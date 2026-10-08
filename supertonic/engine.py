#!/usr/bin/env python
"""
Supertonic TTS engine (plain library — no HTTP framework).

Contract (shared by every engine in this repo):
    load()      load the model once (idempotent)
    run(request) -> dict   synthesize one request
    is_ready()  whether load() has finished

run() request: {"text": "...", "voice": "M1", "lang": "en", "speed": 1.05,
                "steps": 8, "out": "/path/to/out.wav" (optional)}
With "out": writes the WAV there and returns
    {"ok": True, "sample_rate": 44100, "duration_s": F}
Without "out": returns the same dict plus "wav_bytes" (the WAV file contents).
Failures raise EngineError.

Used in-process by vidgen's engines library. server.py is an optional HTTP
wrapper around this module for standalone use.
"""
import io
import logging
from pathlib import Path
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supertonic")

_SAMPLE_RATE = 44_100
NAME = "supertonic"
ROUTE = "/synthesize"
DEFAULT_PORT = 8009


class EngineError(Exception):
    def __init__(self, detail: str, status: int = 500):
        super().__init__(detail)
        self.detail = detail
        self.status = status


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


_state: dict = {}


def load() -> None:
    if "tts" not in _state:
        _state["tts"] = _load_model()


def is_ready() -> bool:
    return "tts" in _state


def run(request: dict) -> dict:
    import soundfile as sf

    load()
    text = request.get("text")
    if not isinstance(text, str) or not text.strip():
        raise EngineError("text is required", 422)
    voice = request.get("voice", "M1")
    lang = request.get("lang", "na")
    speed = float(request.get("speed", 1.05))
    steps = int(request.get("steps", 8))
    out = request.get("out")

    try:
        sample_rate, audio = _synthesise(_state["tts"], text, voice, lang, speed, steps)
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise EngineError(f"{type(exc).__name__}: {exc}") from exc

    duration_s = float(len(audio) / sample_rate)
    result = {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, audio, sample_rate)
        return result
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return {**result, "wav_bytes": buf.getvalue()}
