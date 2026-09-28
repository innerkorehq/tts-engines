#!/usr/bin/env python
"""
k2-fsa/OmniVoice synthesis service.

Loads the OmniVoice model ONCE at startup and serves synthesis requests over
a small HTTP API (FastAPI) for the lifetime of the process.

Endpoints:
    GET  /health       -> {"status": "ok"} once the model has finished loading.
    POST /synthesize    Request body:
                           {"text": "...", "ref_audio": "/abs/ref.wav" (optional),
                            "ref_text": "..." (optional),
                            "out": "/path/to/out.wav" (optional)}
                         If "out" is given, the server writes the WAV file to
                         that path and responds with JSON:
                           {"ok": true, "sample_rate": 24000, "duration_s": F}
                         If "out" is omitted, the response body IS the WAV
                         audio (Content-Type: audio/wav), with sample_rate/
                         duration_s in the X-Sample-Rate/X-Duration-Seconds
                         response headers.
                         On error: an HTTP error status with a JSON
                         {"detail": "..."} body.

All inference behavior/parameters/defaults below are copied verbatim from
the original synth.py — this is a structural refactor (cold-start-per-call
-> warm-service-serves-many), not a behavior change. See the short-text-
degeneration retry logic and the MPS-OOM/CPU-fallback logic below.

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8008)
Env vars: PORT (default 8008)
"""
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omnivoice")

_MODEL_ID = "k2-fsa/OmniVoice"
_DEFAULT_PORT = 8008
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


# ── FastAPI app ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load the primary (MPS-or-CPU) model at startup so /health
    # actually means "ready to serve". The CPU fallback model is
    # intentionally left lazy — it's only ever needed after an MPS OOM.
    logger.info("Loading OmniVoice model (one-time, this may take a while)…")
    _state["model"] = _get_model()
    logger.info("Startup load complete.")
    yield


app = FastAPI(title="omnivoice", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    out: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok" if "model" in _state else "loading"}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    import soundfile as sf

    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model still loading — retry shortly.")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is empty")

    try:
        sample_rate, audio = _synthesise(req.text, req.ref_audio or "", req.ref_text or "")
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    duration_s = len(audio) / sample_rate

    if req.out:
        Path(req.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(req.out, audio, sample_rate)
        return {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Duration-Seconds": f"{duration_s:.3f}",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", _DEFAULT_PORT)))
