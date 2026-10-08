#!/usr/bin/env python
"""
k2-fsa/OmniVoice synthesis service.

Plain library module (no HTTP framework). Contract shared by every engine:
    load()         load the model once (idempotent)
    is_ready()     whether load() has finished
    run(request)   handle one request dict -> result dict; failures raise EngineError
With an "out" path the result is {"ok": True, "sample_rate": ..., "duration_s": ...} after writing
the WAV there; without one, the same dict plus "wav_bytes". vidgen calls this in-process (no HTTP);
server.py is an optional HTTP wrapper for standalone use.

Env vars: PORT (default 8008)
"""
import io
import logging
import os
from dataclasses import dataclass, fields as _dc_fields
from pathlib import Path
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omnivoice")

NAME = "omnivoice"
ROUTE = "/synthesize"
DEFAULT_PORT = 8008


class EngineError(Exception):
    def __init__(self, detail: str, status: int = 500):
        super().__init__(detail)
        self.detail = detail
        self.status = status

_MODEL_ID = "k2-fsa/OmniVoice"
_SAMPLE_RATE = 24000

# Quality gate for the short-text-degeneration issue. Thresholds derived
# from real samples: healthy short clips ("API", "URL") measured rms/peak
# ~0.20; degenerate clips ("Hello", "GitHub" — confirmed gibberish/silent
# via Whisper) measured rms/peak ~0.41-0.49 (continuous loud buzz, no
# pauses, unlike real speech which has bursts/pauses pulling rms well below
# peak).
_MIN_PEAK = 0.01
_MAX_RMS_TO_PEAK_RATIO = 0.35
_MAX_GENERATION_ATTEMPTS = 4

# ── Model state ──────────────────────────────────────────────────────────
_model = None
_cpu_model = None
_force_cpu = False  # sticky once an MPS OOM is hit — MPS is contended for the rest of this run


def _from_pretrained_local_first(model_cls, **kwargs):
    """local_files_only-first to avoid an unnecessary network round-trip
    when the model is already cached. Falls back to a normal (network-
    enabled) load only if nothing is cached yet."""
    try:
        return model_cls.from_pretrained(_MODEL_ID, local_files_only=True, **kwargs)
    except Exception:
        logger.info("Not fully cached locally yet — downloading %s…", _MODEL_ID)
        return model_cls.from_pretrained(_MODEL_ID, **kwargs)


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


def _is_mps_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _generate(kwargs: dict):
    """Run model.generate(), falling back to a CPU copy of the model on an
    MPS out-of-memory error instead of failing. `_force_cpu` is
    process-lifetime state, so once tripped, every subsequent request uses
    the CPU model."""
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


def _synthesise(text: str, ref_audio: str = "", ref_text: str = ""):
    """Returns a (sample_rate, mono float32 PCM ndarray) tuple."""
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
            "Generation %d produced degenerate audio (known short-text instability); "
            "retrying (attempt %d/%d)…",
            attempt - 1, attempt, _MAX_GENERATION_ATTEMPTS,
        )
        audio = _generate(kwargs)

    return _SAMPLE_RATE, audio


@dataclass
class SynthesizeRequest:
    text: str
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    out: Optional[str] = None


_state: dict = {}
_loaded = False


def load() -> None:
    global _loaded
    if _loaded:
        return
    # Eagerly load the primary (MPS-or-CPU) model at startup so /health
    # actually means "ready to serve". The CPU fallback model is
    # intentionally left lazy — it's only ever needed after an MPS OOM.
    logger.info("Loading OmniVoice model (one-time, this may take a while)…")
    _state["model"] = _get_model()
    logger.info("Startup load complete.")
    _loaded = True


def is_ready() -> bool:
    return _loaded


def run(request: dict) -> dict:
    load()
    try:
        fields_ = {k: v for k, v in request.items() if k in {f.name for f in _dc_fields(SynthesizeRequest)}}
        req = SynthesizeRequest(**fields_)
    except TypeError as exc:
        raise EngineError(f"invalid request: {exc}", 422) from exc
    import soundfile as sf

    if "model" not in _state:
        raise EngineError("Model still loading — retry shortly.", 503)
    if not req.text.strip():
        raise EngineError("text is empty", 422)

    try:
        sample_rate, audio = _synthesise(req.text, req.ref_audio or "", req.ref_text or "")
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise EngineError(f"{type(exc).__name__}: {exc}", 500) from exc

    duration_s = len(audio) / sample_rate

    if req.out:
        Path(req.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(req.out, audio, sample_rate)
        return {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return {"ok": True, "sample_rate": sample_rate, "duration_s": float(duration_s), "wav_bytes": buf.getvalue()}
