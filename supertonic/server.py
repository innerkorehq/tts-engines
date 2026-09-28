#!/usr/bin/env python
"""
Supertonic TTS synthesis service.

Loads the Supertonic ONNX model ONCE at startup and serves synthesis
requests over a small HTTP API (FastAPI) for the lifetime of the process.

Endpoints:
    GET  /health       -> {"status": "ok"} once the model has finished loading.
    POST /synthesize    Request body:
                           {"text": "...", "voice": "M1", "lang": "en",
                            "speed": 1.05, "steps": 8,
                            "out": "/path/to/out.wav" (optional)}
                         If "out" is given, the server writes the WAV file to
                         that path and responds with JSON:
                           {"ok": true, "sample_rate": 44100, "duration_s": F}
                         If "out" is omitted, the response body IS the WAV
                         audio (Content-Type: audio/wav), with sample_rate/
                         duration_s in the X-Sample-Rate/X-Duration-Seconds
                         response headers.
                         On error: an HTTP error status with a JSON
                         {"detail": "..."} body.

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8009)
Env vars: PORT (default 8009)
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
logger = logging.getLogger("supertonic")

_SAMPLE_RATE = 44_100
_DEFAULT_PORT = 8009


def _load_model():
    # Loads the ONNX session(s) from disk, downloading model files first if
    # needed.
    logger.info("Loading Supertonic TTS 3 (one-time, model download may occur)…")
    from supertonic import TTS
    tts = TTS(auto_download=True)
    logger.info("Supertonic TTS 3 ready.")
    return tts


def _synthesise(tts, text: str, voice: str, lang: str, speed: float, steps: int):
    """Returns a (sample_rate, mono float32 PCM ndarray) tuple.

    `get_voice_style` re-reads the style file per call (no shared mutable
    cache to worry about); `synthesize` only touches `tts.model`, the
    already-loaded ONNX session — safe to reuse across many calls, which is
    the whole point of loading it once.
    """
    style = tts.get_voice_style(voice_name=voice)
    wav, duration = tts.synthesize(
        text=text,
        lang=lang,
        voice_style=style,
        total_steps=steps,
        speed=speed,
    )
    # wav: float32 numpy array (1, num_samples); trim to actual duration
    samples = wav[0, : int(_SAMPLE_RATE * duration[0].item())]
    return _SAMPLE_RATE, samples


# ── FastAPI app ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["tts"] = _load_model()
    yield


app = FastAPI(title="supertonic", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "M1"
    lang: str = "na"
    speed: float = 1.05
    steps: int = 8
    out: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok" if "tts" in _state else "loading"}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    import soundfile as sf

    tts = _state.get("tts")
    if tts is None:
        raise HTTPException(status_code=503, detail="Model still loading — retry shortly.")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is required")

    try:
        sample_rate, audio = _synthesise(tts, req.text, req.voice, req.lang, req.speed, req.steps)
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    duration_s = float(len(audio) / sample_rate)

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
