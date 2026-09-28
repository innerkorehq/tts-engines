#!/usr/bin/env python
"""
Chatterbox-Multilingual-hi (gagan1985/chatterbox-multilingual-hi-mlx-fp16)
persistent synthesis daemon.

The original `synth.py` is invoked as a *fresh* subprocess per scene, so the
mlx-audio Chatterbox model (T3 + voice-encoder + s3gen decoder) gets loaded
from disk from scratch on every single narration clip. This script instead
loads the model ONCE and then serves synthesis requests over a simple
line-delimited JSON protocol on stdin/stdout, kept alive for the lifetime of
the worker process — mirroring `tts-engines/f5tts/f5tts_server.py`.

Protocol (one JSON object per line, UTF-8, newline-terminated):
    Request  -> {"text": "...", "ref_audio": "/path/to/ref.wav", "out": "/path/to/out.wav",
                 "voice_id": "vp_..." (optional)}
    Response <- {"ok": true, "sample_rate": N, "duration_s": F}
             <- {"ok": false, "error": "..."}
"voice_id" is the stable voice_profiles.id — when present, the expensive
ref_audio -> speaker-conditioning pass is cached on disk keyed by it (see
the speaker-conditioning cache section below), so repeat requests for the
same voice skip straight to generation. Omitting it falls back to the
original always-recompute-from-ref_audio path.
A "READY" line is written to the protocol stdout once the model has finished
loading, so the parent process knows when it's safe to start sending
requests. A "SHUTDOWN" line (not JSON) exits cleanly.

This is a structural refactor only (cold-start-per-call -> warm-daemon) —
inference behavior/parameters/defaults are unchanged from synth.py.
"""
import json
import os
import sys
import traceback
from pathlib import Path

# ── Protect the protocol stream from library noise ───────────────────────────
# mlx_audio (and its dependencies) call bare `print(...)` during model loading
# AND during inference, which would land on stdout and corrupt our
# line-delimited JSON protocol if left alone. So: duplicate the *original*
# stdout fd into a dedicated file object reserved exclusively for protocol
# messages (READY / JSON responses), then repoint `sys.stdout` at stderr so
# every incidental `print()` from imported libraries is harmless diagnostic
# noise instead of protocol corruption. This must happen BEFORE importing any
# ML libraries below.
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


import logging

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("chatterbox-multilingual-hi-server")

_REPO = "gagan1985/chatterbox-multilingual-hi-mlx-fp16"
_LANGUAGE_ID = "hi"
_TARGET_SR = 24_000

_model = None

# ── Speaker-conditioning cache ────────────────────────────────────────────
# model.prepare_conditionals(ref_wav, ...) runs the S3 tokenizer twice,
# s3gen.embed_ref, and the voice encoder — real forward passes, not free —
# to turn a ref_audio clip into a Conditionals(t3, gen) object that
# model.generate(conds=...) can reuse directly, skipping all of that. Voice
# profiles are stable/named (voice_profiles.id, threaded through as
# "voice_id" in the request), so we cache the computed Conditionals per
# voice_id: an in-memory dict for same-process reuse, backed by a
# conds.safetensors-format file on disk (the same format
# Model.from_pretrained already knows how to read — see chatterbox.py) so
# the cache survives this daemon being killed/restarted by the cross-process
# heavy-model slot. A cheap (size, mtime) fingerprint of ref_audio, stored in
# the safetensors file's own metadata, invalidates the cache if the
# underlying reference clip is ever replaced under the same voice_id.
_CONDS_CACHE_DIR = Path(__file__).parent / ".cache" / "speaker_conds"
_conds_mem_cache: dict[str, tuple] = {}  # voice_id -> (fingerprint, Conditionals)


def _log(msg: str) -> None:
    """Diagnostic logging to stderr — stdout is reserved for the protocol."""
    print(f"[chatterbox-multilingual-hi-server] {msg}", file=sys.stderr, flush=True)


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or anything that imports
    it, e.g. mlx_audio below) before HF_HUB_OFFLINE is set makes the env
    var a no-op (huggingface_hub reads it into a module-level constant at
    import time, not dynamically per call).
    """
    from pathlib import Path

    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _load_model():
    """One-time setup: resolve offline-cache mode and load the mlx-audio
    Chatterbox model. Runs once at daemon startup, before the READY signal."""
    global _model

    # Every call to this engine used to run in a brand-new subprocess — there
    # was no in-process cache that survived between calls, so without forcing
    # offline mode, EVERY synthesis call re-resolved every checkpoint file
    # against huggingface.co over the network, even though it was already
    # cached locally. mlx_audio.tts.utils.load_model() doesn't expose a
    # local_files_only-style kwarg, so HF_HUB_OFFLINE forces the same
    # behavior transparently — but it MUST be set before importing
    # mlx_audio (it's read into a module-level constant at import time, not
    # rechecked per call), so we check the cache directory directly first
    # rather than importing mlx_audio in a try/except. In the daemon this
    # only needs to happen once, at startup.
    if _repo_looks_cached(_REPO):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        _log(f"Not fully cached locally yet — downloading {_REPO}…")

    # mlx_audio also auto-downloads the shared S3TokenizerV2 weights it
    # depends on (mlx-community/S3TokenizerV2) on first run — left to its
    # own default resolution, same as the mlx-audio engine's other models.
    if not os.environ.get("HF_TOKEN"):
        os.environ.pop("HF_TOKEN", None)

    from mlx_audio.tts.utils import load_model

    _log(f"Loading {_REPO} (one-time, this may take a while)…")
    _model = load_model(_REPO)
    _log("Chatterbox-Multilingual-hi ready.")
    return _model


def _ref_audio_fingerprint(ref_audio: str) -> str:
    """Cheap (size, mtime) fingerprint — enough to detect the reference clip
    being replaced under the same voice_id without hashing file contents."""
    st = os.stat(ref_audio)
    return f"{st.st_size}:{int(st.st_mtime)}"


def _conds_to_flat_dict(conds) -> dict:
    """Flatten a Conditionals(t3, gen) object into the same flat key format
    Model.from_pretrained's conds.safetensors loader expects (chatterbox.py
    ~line 611), so the cache file is readable by that existing loader too."""
    flat = {
        "t3.speaker_emb": conds.t3.speaker_emb,
        "t3.emotion_adv": conds.t3.emotion_adv,
    }
    if conds.t3.cond_prompt_speech_tokens is not None:
        flat["t3.cond_prompt_speech_tokens"] = conds.t3.cond_prompt_speech_tokens
    for k, v in conds.gen.items():
        flat[f"gen.{k}"] = v
    return flat


def _flat_dict_to_conds(flat: dict):
    """Inverse of _conds_to_flat_dict — mirrors Model.from_pretrained's own
    conds.safetensors parsing (chatterbox.py ~line 611-644)."""
    from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals, T3Cond

    t3_cond = T3Cond(
        speaker_emb=flat["t3.speaker_emb"],
        cond_prompt_speech_tokens=flat.get("t3.cond_prompt_speech_tokens"),
        emotion_adv=flat["t3.emotion_adv"],
    )
    gen_dict = {k[len("gen."):]: v for k, v in flat.items() if k.startswith("gen.")}
    return Conditionals(t3_cond, gen_dict)


def _load_or_build_conds(voice_id: str, ref_audio: str):
    """Same-process mem cache -> on-disk cache (validated against a cheap
    ref_audio fingerprint) -> compute via model.prepare_conditionals(),
    writing through to both tiers. A corrupt/stale/missing cache entry is
    never fatal — it just falls back to recomputing."""
    fingerprint = _ref_audio_fingerprint(ref_audio)

    cached = _conds_mem_cache.get(voice_id)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    cache_path = _CONDS_CACHE_DIR / f"{voice_id}.safetensors"
    if cache_path.exists():
        try:
            import mlx.core as mx

            flat, metadata = mx.load(str(cache_path), return_metadata=True)
            if metadata.get("fingerprint") == fingerprint:
                conds = _flat_dict_to_conds(flat)
                _conds_mem_cache[voice_id] = (fingerprint, conds)
                _log(f"Speaker-conditioning cache hit for voice_id={voice_id!r} (disk).")
                return conds
            _log(f"Speaker-conditioning cache stale for voice_id={voice_id!r} — ref_audio changed.")
        except Exception as exc:
            _log(f"Speaker-conditioning cache unreadable for voice_id={voice_id!r} ({exc}) — recomputing.")

    _log(f"Speaker-conditioning cache miss for voice_id={voice_id!r} — computing.")
    conds = _model.prepare_conditionals(ref_audio, _TARGET_SR)
    _conds_mem_cache[voice_id] = (fingerprint, conds)

    try:
        import mlx.core as mx

        _CONDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(cache_path), _conds_to_flat_dict(conds), metadata={"fingerprint": fingerprint},
        )
    except Exception as exc:
        _log(f"Failed to write speaker-conditioning cache for voice_id={voice_id!r} ({exc}) — continuing without it.")

    return conds


def _handle_request(req: dict) -> dict:
    """Per-request work: the actual synthesis call, using text/ref_audio/out
    from each request. Runs once per JSON line received."""
    import numpy as np
    import soundfile as sf

    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    voice_id = req.get("voice_id")
    out = req["out"]

    kwargs: dict = {"text": text, "lang_code": _LANGUAGE_ID}
    if ref_audio and voice_id:
        kwargs["conds"] = _load_or_build_conds(voice_id, ref_audio)
    elif ref_audio:
        kwargs["ref_audio"] = ref_audio
    results = list(_model.generate(**kwargs))
    if not results:
        raise RuntimeError("Chatterbox-Multilingual-hi produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out, audio, sample_rate)

    return {"ok": True, "sample_rate": sample_rate, "duration_s": len(audio) / sample_rate}


def main() -> int:
    _load_model()

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
