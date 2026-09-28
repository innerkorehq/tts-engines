#!/usr/bin/env python
"""
k2-fsa/OmniVoice persistent synthesis daemon.

`synth.py` (kept as-is, used as a one-shot fallback if this daemon fails to
start) is invoked fresh for every scene — meaning the OmniVoice model gets
reloaded onto MPS/CPU from scratch on every single narration clip. For a
render with N scenes that's N full model loads instead of one.

This script instead loads the model ONCE and then serves synthesis requests
over a simple line-delimited JSON protocol on stdin/stdout, kept alive for the
lifetime of the worker process — same pattern as `tts-engines/f5tts/f5tts_server.py`.

Protocol (one JSON object per line, UTF-8, newline-terminated):
    Request  -> {"text": "...", "out": "/abs/out.wav",
                 "ref_audio": "/abs/ref.wav" (optional), "ref_text": "..." (optional)}
    Response <- {"ok": true, "sample_rate": 24000, "duration_s": F}
             <- {"ok": false, "error": "..."}
A "READY" line is written to the protocol stdout once the model has finished
loading, so the parent process knows when it's safe to start sending requests.
A "SHUTDOWN" line (exact string, not JSON) exits cleanly.

All inference behavior/parameters/defaults below are copied verbatim from
`synth.py` — this is a structural refactor (cold-start-per-call ->
warm-daemon-serves-many), not a behavior change. See that file's module
docstring for the full rationale behind the short-text-degeneration retry
logic and the MPS-OOM/CPU-fallback logic reproduced here.
"""
import json
import logging
import os
import sys
import traceback
from pathlib import Path

# ── Protect the protocol stream from library noise ───────────────────────────
# OmniVoice's own code (and/or its dependencies, e.g. torch/transformers) may
# call bare `print(...)` during model loading and/or inference, which would
# land on stdout and corrupt our line-delimited JSON protocol if left alone.
# So: duplicate the *original* stdout fd into a dedicated file object reserved
# exclusively for protocol messages (READY / JSON responses), then repoint
# `sys.stdout` at stderr so every incidental `print()` from imported libraries
# is harmless diagnostic noise instead of protocol corruption. This must
# happen BEFORE any ML libraries are imported.
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("omnivoice_server")

_MODEL_ID = "k2-fsa/OmniVoice"

# Quality gate for the short-text-degeneration issue (see synth.py's module
# docstring). Thresholds derived from real samples: healthy short clips
# ("API", "URL") measured rms/peak ~0.20; degenerate clips ("Hello", "GitHub"
# — confirmed gibberish/silent via Whisper) measured rms/peak ~0.41-0.49
# (continuous loud buzz, no pauses, unlike real speech which has
# bursts/pauses pulling rms well below peak).
_MIN_PEAK = 0.01
_MAX_RMS_TO_PEAK_RATIO = 0.35
_MAX_GENERATION_ATTEMPTS = 4

# ── One-time model state (module scope so it naturally persists across every
# request for the lifetime of this daemon process) ───────────────────────────
_model = None
_cpu_model = None
_force_cpu = False  # sticky once an MPS OOM is hit — MPS is contended for the rest of this run


def _get_model():
    global _model
    if _model is not None:
        return _model

    import torch
    from omnivoice import OmniVoice

    device_map = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device_map == "mps" else torch.float32
    logger.info("Loading %s (device_map=%s, dtype=%s)…", _MODEL_ID, device_map, dtype)
    _model = _from_pretrained_local_first(OmniVoice, device_map=device_map, dtype=dtype)
    logger.info("OmniVoice ready.")
    return _model


def _get_cpu_model():
    global _cpu_model
    if _cpu_model is not None:
        return _cpu_model

    import torch
    from omnivoice import OmniVoice

    logger.info("Loading %s (device_map=cpu fallback)…", _MODEL_ID)
    _cpu_model = _from_pretrained_local_first(OmniVoice, device_map="cpu", dtype=torch.float32)
    logger.info("OmniVoice (CPU fallback) ready.")
    return _cpu_model


def _from_pretrained_local_first(model_cls, **kwargs):
    """
    Unlike the one-shot synth.py (a brand-new subprocess per call, hence
    local_files_only to avoid re-resolving every file over the network each
    time), this daemon only ever calls this once per model per process
    lifetime — but local_files_only-first is kept anyway since it's still
    correct and avoids an unnecessary network round-trip on the (only) load.
    Falls back to a normal (network-enabled) load only if nothing is cached
    yet.
    """
    try:
        return model_cls.from_pretrained(_MODEL_ID, local_files_only=True, **kwargs)
    except Exception:
        logger.info("Not fully cached locally yet — downloading %s…", _MODEL_ID)
        return model_cls.from_pretrained(_MODEL_ID, **kwargs)


def _is_mps_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _generate(kwargs: dict):
    """Run model.generate(), falling back to a CPU copy of the model on an
    MPS out-of-memory error (see synth.py's module docstring) instead of
    failing. `_force_cpu` is process-lifetime state, so once tripped, every
    subsequent request in this daemon uses the CPU model."""
    global _force_cpu

    if not _force_cpu:
        try:
            model = _get_model()
            return model.generate(**kwargs)[0]
        except RuntimeError as e:
            if not _is_mps_oom(e):
                raise
            import torch
            logger.warning("MPS out of memory (%s) — falling back to CPU for this and remaining calls", e)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            _force_cpu = True

    return _get_cpu_model().generate(**kwargs)[0]


def _is_degenerate(audio) -> bool:
    import numpy as np

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < _MIN_PEAK:
        return True
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return (rms / peak) > _MAX_RMS_TO_PEAK_RATIO


def synthesise(text: str, out_path: str, ref_audio: str = "", ref_text: str = "") -> tuple[int, float]:
    import soundfile as sf

    sample_rate = 24000

    kwargs: dict = {"text": text}
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
        if ref_text:
            kwargs["ref_text"] = ref_text
        else:
            logger.info("ref_audio given without ref_text — model will auto-transcribe via Whisper")

    audio = _generate(kwargs)
    attempt = 1
    while _is_degenerate(audio) and attempt < _MAX_GENERATION_ATTEMPTS:
        attempt += 1
        logger.warning(
            "Generation %d produced degenerate audio (known short-text instability — "
            "see synth.py's module docstring); retrying (attempt %d/%d)…",
            attempt - 1, attempt, _MAX_GENERATION_ATTEMPTS,
        )
        audio = _generate(kwargs)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


def _load_models():
    # Eagerly load the primary (MPS-or-CPU) model at startup, before READY is
    # sent, so READY actually means "ready to serve" rather than "ready to
    # lazily load on first request". The CPU fallback model is intentionally
    # left lazy (_get_cpu_model) — it's only ever needed after an MPS OOM.
    logger.info("Loading OmniVoice model (one-time, this may take a while)…")
    _get_model()
    logger.info("Startup load complete.")


def _handle_request(req: dict) -> dict:
    text = req.get("text", "")
    out_path = req.get("out")
    ref_audio = req.get("ref_audio", "")
    ref_text = req.get("ref_text", "")

    if not text.strip():
        raise ValueError("text is empty")
    if not out_path:
        raise ValueError("out path is required")

    sample_rate, duration_s = synthesise(text, out_path, ref_audio, ref_text)
    return {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}


def main() -> int:
    _load_models()

    # Signal readiness to the parent process on the protected protocol stream
    # (NOT via `print`, which now points at stderr).
    _send("READY")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "SHUTDOWN":
            logger.info("Received SHUTDOWN — exiting.")
            return 0
        try:
            req = json.loads(line)
            resp = _handle_request(req)
        except Exception as exc:  # noqa: BLE001 — must always answer the request
            logger.error("Error handling request: %s\n%s", exc, traceback.format_exc())
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(json.dumps(resp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
