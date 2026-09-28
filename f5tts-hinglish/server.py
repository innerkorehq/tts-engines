"""
F5-TTS-Hinglish persistent synthesis daemon.

`synth.py` in this directory is invoked fresh for every narration clip —
meaning the F5-TTS-Small DiT checkpoint (fine-tuned for Hindi-English
code-switched narration, `rajputsw/F5-TTS-Hinglish`) gets downloaded/loaded
from scratch on every single call. For a render with N scenes that's N full
model loads instead of one.

This script instead loads the model ONCE and then serves synthesis requests
over a simple line-delimited JSON protocol on stdin/stdout, kept alive for the
lifetime of the worker process — mirroring the pattern used by the sibling
`tts-engines/f5tts/f5tts_server.py` daemon for the stock F5-TTS engine.

Protocol (one JSON object per line, UTF-8, newline-terminated) — same request/
response shape as the one-shot `synth.py` CLI (see `protocol.read_request` /
`protocol.succeed` / `protocol.fail`):
    Request  -> {"text": "...", "ref_audio": "...", "ref_text": "...", "out": "..."}
    Response <- {"ok": true, "sample_rate": 24000, "duration_s": 3.21}
             <- {"ok": false, "error": "..."}
A "READY" line is written to the protocol stdout stream once the model has
finished loading, so the parent process knows when it's safe to start sending
requests. A "SHUTDOWN" line (bare string, not JSON) exits cleanly.

Why the ThreadPoolExecutor patch: F5-TTS's own `infer_process`/
`infer_batch_process` (used internally by `f5_tts.api.F5TTS.infer`) spins up a
`ThreadPoolExecutor` that isn't safe for concurrent `model_obj.sample()` calls
on Apple Silicon MPS — patched to max_workers=1 below, same as the stock
f5tts_server.py daemon, and for the same reason.
"""
import concurrent.futures as _cf

_OrigTPE = _cf.ThreadPoolExecutor


class _SequentialTPE(_OrigTPE):
    """Drop-in replacement that processes tasks one at a time (MPS isn't
    thread-safe for concurrent model_obj.sample() calls)."""

    def __init__(self, *args, **kwargs):
        kwargs["max_workers"] = 1
        super().__init__(*args, **kwargs)


_cf.ThreadPoolExecutor = _SequentialTPE  # patch before f5_tts imports

import json
import os
import sys
import traceback
from pathlib import Path

# ── Protect the protocol stream from library noise ───────────────────────────
# F5-TTS's own code (and huggingface_hub's download progress) calls bare
# `print(...)` during model loading AND during inference — all of which would
# land on stdout and corrupt our line-delimited JSON protocol if left alone.
# So: duplicate the *original* stdout fd into a dedicated file object reserved
# exclusively for protocol messages (READY / JSON responses), then repoint
# `sys.stdout` at stderr so every incidental `print()` from imported libraries
# is harmless diagnostic noise instead of protocol corruption. This must
# happen BEFORE importing huggingface_hub/f5_tts/soundfile.
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


import logging

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("f5tts-hinglish-server")

_HF_REPO = "rajputsw/F5-TTS-Hinglish"
_MODEL_CONFIG = "F5TTS_Small"  # 768-dim/18-layer/12-head DiT, matches this checkpoint


def _log(msg: str) -> None:
    """Diagnostic logging to stderr — stdout is reserved for the protocol."""
    print(f"[f5tts-hinglish-server] {msg}", file=sys.stderr, flush=True)


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it before HF_HUB_OFFLINE is set
    makes the env var a no-op (huggingface_hub reads it into a
    module-level constant at import time, not dynamically per call). Same
    pattern as chatterbox-multilingual-hi/server.py's identical helper.
    """
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _load_model():
    """One-time setup: download checkpoint/vocab and load the F5TTS model.

    Runs once at daemon startup, before the READY signal. Mirrors exactly
    what synth.py did per-call, just hoisted out of the per-request path.
    """
    # hf_hub_download() hits huggingface.co's API unconditionally (no local
    # cache check of its own) unless HF_HUB_OFFLINE is set — force it once
    # this repo is confirmed cached, so a transient HF Hub connectivity blip
    # can't crash startup for a checkpoint that's already on disk (confirmed
    # in production for the sibling local-chat-mlx daemon).
    if _repo_looks_cached(_HF_REPO):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        logger.info("Not fully cached locally yet — downloading %s…", _HF_REPO)

    from huggingface_hub import hf_hub_download
    from f5_tts.api import F5TTS

    logger.info("Fetching %s checkpoint/vocab from Hugging Face Hub…", _HF_REPO)
    ckpt_file = hf_hub_download(repo_id=_HF_REPO, filename="model_last.pt")
    vocab_file = hf_hub_download(repo_id=_HF_REPO, filename="vocab.txt")

    logger.info("Loading F5-TTS-Hinglish (%s, config=%s)…", _HF_REPO, _MODEL_CONFIG)
    tts = F5TTS(model=_MODEL_CONFIG, ckpt_file=ckpt_file, vocab_file=vocab_file)
    logger.info("F5-TTS-Hinglish model ready on device=%s", tts.device)
    return tts


def _handle_request(req: dict, tts) -> dict:
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    out = req["out"]

    wav, sample_rate, _spec = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        show_info=lambda *a, **k: None,
        remove_silence=True,
        # f5_tts.api.F5TTS.infer(seed=None) does
        # `random.randint(0, sys.maxsize)`, which is almost always out of
        # PYTHONHASHSEED's valid [0, 4294967295] range and crashes any forked
        # worker with "Fatal Python error: config_init_hash_seed". Pass an
        # in-range seed explicitly to avoid that.
        seed=0,
    )

    import soundfile as sf
    sf.write(out, wav, sample_rate)

    return {"ok": True, "sample_rate": sample_rate, "duration_s": len(wav) / sample_rate}


def main() -> int:
    tts = _load_model()

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
            resp = _handle_request(req, tts)
        except Exception as exc:  # noqa: BLE001 — must always answer the request
            _log(f"Error handling request: {exc}\n{traceback.format_exc()}")
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(json.dumps(resp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
