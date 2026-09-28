#!/usr/bin/env python
"""
hinglish-tts persistent synthesis daemon.

The original `synth.py` wrapper is invoked fresh (per scene) via vidgen's
stdin/stdout JSON subprocess contract (see protocol.py) — meaning the
IndicF5 model loaded by `inference.load_model()` gets reloaded from scratch
on every single narration clip. For a render with N scenes that's N full
model loads instead of one.

This script instead loads the model ONCE and then serves synthesis requests
over a simple line-delimited JSON protocol on stdin/stdout, kept alive for
the lifetime of the worker process. The caller (src/infra/adapters/tts.py)
talks to it as a long-running daemon (spawning it lazily on first use and
auto-restarting it if it crashes), instead of shelling out per call.

Protocol (one JSON object per line, UTF-8, newline-terminated):
    Request  -> {"text": "...", "ref_audio": "...", "ref_text": "...", "out": "..."}
    Response <- {"ok": true, "sample_rate": 24000, "duration_s": F}
             <- {"ok": false, "error": "..."}
A "READY" line is written to the protocol stream once the model has finished
loading, so the parent process knows when it's safe to start sending
requests. A "SHUTDOWN" line (exact string, not JSON) exits cleanly.

No reimplementation — inference.py and scoring/scripts/lib_normalize.py are
vendored verbatim from the upstream repo (harrrshall/hinglish-tts); this
file only adapts them to a warm-daemon-serves-many structure, same as
tts-engines/f5tts/f5tts_server.py does for F5-TTS. synth.py stays as-is,
used as a one-shot fallback by the caller if the daemon fails to start.
"""
import json
import os
import sys
import traceback
from pathlib import Path

# ── Protect the protocol stream from library noise ───────────────────────────
# inference.py (vendored upstream) calls bare `print(...)` during its
# duration-patch diagnostics (see the "DEBUG-PATCH: ref_chars=..." print in
# inference.py) and possibly elsewhere during model load / synthesis — all of
# which would land on stdout and corrupt our line-delimited JSON protocol if
# left alone. So: duplicate the *original* stdout fd into a dedicated file
# object reserved exclusively for protocol messages (READY / JSON responses),
# then repoint `sys.stdout` at stderr so every incidental `print()` from
# imported libraries is harmless diagnostic noise instead of protocol
# corruption. This must happen BEFORE importing any of the ML/vendored code.
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


import logging

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("hinglish-tts-server")


def _log(msg: str) -> None:
    """Diagnostic logging to stderr — stdout is reserved for the protocol."""
    print(f"[hinglish-tts-server] {msg}", file=sys.stderr, flush=True)


def _apply_fairseq_compat_shims() -> None:
    """
    fairseq==0.12.2 (pulled in by ai4bharat-transliteration, used by the
    vendored scoring/scripts/lib_normalize.py) needs two scoped compat
    shims on Python 3.11 / torch>=2.6 — same as tts-engines/indic-xlit/
    synth.py's _load_xlit_engine, and identical to synth.py's own helper of
    the same name. Applied here, once, before any inference — the vendored
    upstream files themselves are left untouched.
    """
    import dataclasses as _dataclasses
    import torch as _torch

    orig_get_field = _dataclasses._get_field

    def _get_field_compat(cls_, a_name, a_type, default_kw_only, *args, **kwargs):
        default = getattr(cls_, a_name, _dataclasses.MISSING)
        target = default.default if isinstance(default, _dataclasses.Field) else default
        if target is not _dataclasses.MISSING and target.__class__.__hash__ is None:
            try:
                target.__class__.__hash__ = object.__hash__
            except TypeError:
                pass
        return orig_get_field(cls_, a_name, a_type, default_kw_only, *args, **kwargs)

    orig_torch_load = _torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    _dataclasses._get_field = _get_field_compat
    _torch.load = _torch_load_compat


_apply_fairseq_compat_shims()


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or anything that imports it,
    e.g. inference.py below, which loads ai4bharat/IndicF5 via plain
    transformers.AutoModel.from_pretrained with no local-cache check of its
    own) before HF_HUB_OFFLINE is set makes the env var a no-op
    (huggingface_hub reads it into a module-level constant at import time,
    not dynamically per call). Same pattern as
    chatterbox-multilingual-hi/server.py's identical helper.
    """
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


_REPO = "ai4bharat/IndicF5"
if _repo_looks_cached(_REPO):
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    _log(f"Not fully cached locally yet — downloading {_REPO}…")

from inference import load_model, synthesize  # noqa: E402 — must follow compat shims

# ── One-time model load, memoised at module scope ────────────────────────────
# Mirrors synth.py's `_model` module-global cache — in the daemon this now
# naturally persists across every request for the lifetime of the process,
# since the whole interpreter stays alive instead of exiting after one call.
_model = None


def _get_model():
    global _model
    if _model is None:
        _log("Loading IndicF5 via inference.load_model()…")
        _model = load_model()
        _log("IndicF5 ready.")
    return _model


def _handle_request(req: dict) -> dict:
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    out = req["out"]

    import soundfile as sf

    model = _get_model()
    audio = synthesize(model, text, ref_audio_path=ref_audio, ref_text=ref_text)
    sample_rate = 24000
    sf.write(out, audio, samplerate=sample_rate)

    return {"ok": True, "sample_rate": sample_rate, "duration_s": len(audio) / sample_rate}


def main() -> int:
    # Warm the model once, before entering the request loop, so the READY
    # signal genuinely means "ready to synthesize" (matches f5tts_server.py's
    # behavior of loading before sending READY).
    _get_model()

    # Signal readiness to the parent process on the protected protocol stream
    # (NOT via `print`, which now points at stderr).
    _send("READY")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "SHUTDOWN":
            _log("Received SHUTDOWN — exiting.")
            return 0
        try:
            req = json.loads(line)
            resp = _handle_request(req)
        except Exception as exc:  # noqa: BLE001 — must always answer the request
            _log(f"Error handling request: {exc}\n{traceback.format_exc()}")
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(json.dumps(resp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
