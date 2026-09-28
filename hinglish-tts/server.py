#!/usr/bin/env python
"""
hinglish-tts synthesis service.

Loads the IndicF5 model (via the vendored `inference.py`, upstream:
harrrshall/hinglish-tts) ONCE at startup and serves synthesis requests over
a small HTTP API (FastAPI) for the lifetime of the process.

No reimplementation — `inference.py` and `scoring/scripts/lib_normalize.py`
are vendored verbatim from the upstream repo; this file only adapts them to
a warm-service-serves-many structure.

Endpoints:
    GET  /health       -> {"status": "ok"} once the model has finished loading.
    POST /synthesize    Request body:
                           {"text": "...", "ref_audio": "...", "ref_text": "...",
                            "out": "/path/to/out.wav" (optional)}
                         If "out" is given, the server writes the WAV file to
                         that path and responds with JSON:
                           {"ok": true, "sample_rate": 24000, "duration_s": F}
                         If "out" is omitted, the response body IS the WAV
                         audio (Content-Type: audio/wav).
                         On error: an HTTP error status with a JSON
                         {"detail": "..."} body.

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8005)
Env vars: PORT (default 8005), HF_HOME, HF_TOKEN
"""
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hinglish-tts")

_DEFAULT_PORT = 8005


def _apply_fairseq_compat_shims() -> None:
    """
    fairseq==0.12.2 (pulled in by ai4bharat-transliteration, used by the
    vendored scoring/scripts/lib_normalize.py) needs two scoped compat
    shims on Python 3.11 / torch>=2.6. Applied here, once, before any
    inference — the vendored upstream files themselves are left untouched.
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
    own) before HF_HUB_OFFLINE is set makes the env var a no-op.
    """
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


_REPO = "ai4bharat/IndicF5"
if _repo_looks_cached(_REPO):
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    logger.info("Not fully cached locally yet — downloading %s…", _REPO)

from inference import load_model, synthesize  # noqa: E402 — must follow compat shims
from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

_SAMPLE_RATE = 24_000

# ── FastAPI app ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading IndicF5 via inference.load_model()…")
    _state["model"] = load_model()
    logger.info("IndicF5 ready.")
    yield


app = FastAPI(title="hinglish-tts", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    ref_audio: str
    ref_text: str
    out: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok" if "model" in _state else "loading"}


@app.post("/synthesize")
def synthesize_route(req: SynthesizeRequest):
    import soundfile as sf

    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading — retry shortly.")

    try:
        audio = synthesize(model, req.text, ref_audio_path=req.ref_audio, ref_text=req.ref_text)
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    duration_s = len(audio) / _SAMPLE_RATE

    if req.out:
        Path(req.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(req.out, audio, samplerate=_SAMPLE_RATE)
        return {"ok": True, "sample_rate": _SAMPLE_RATE, "duration_s": duration_s}

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=_SAMPLE_RATE, format="WAV")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(_SAMPLE_RATE),
            "X-Duration-Seconds": f"{duration_s:.3f}",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", _DEFAULT_PORT)))
