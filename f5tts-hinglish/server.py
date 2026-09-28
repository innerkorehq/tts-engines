"""
F5-TTS-Hinglish synthesis service.

Loads the `rajputsw/F5-TTS-Hinglish` DiT checkpoint (fine-tuned for
Hindi-English code-switched narration) ONCE at startup and serves synthesis
requests over a small HTTP API (FastAPI) for the lifetime of the process.

Endpoints:
    GET  /health       -> {"status": "ok"} once the model has finished loading.
    POST /synthesize    Request body:
                           {"text": "...", "ref_audio": "...", "ref_text": "...",
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

Why the ThreadPoolExecutor patch: F5-TTS's own `infer_process`/
`infer_batch_process` (used internally by `f5_tts.api.F5TTS.infer`) spins up
a `ThreadPoolExecutor` that isn't safe for concurrent `model_obj.sample()`
calls on Apple Silicon MPS — patched to max_workers=1 below.

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8004)
Env vars: PORT (default 8004), HF_HOME, HF_TOKEN
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

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("f5tts-hinglish")

_HF_REPO = "rajputsw/F5-TTS-Hinglish"
_MODEL_CONFIG = "F5TTS_Small"  # 768-dim/18-layer/12-head DiT, matches this checkpoint
_DEFAULT_PORT = 8004


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it before HF_HUB_OFFLINE is set
    makes the env var a no-op (huggingface_hub reads it into a module-level
    constant at import time, not dynamically per call).
    """
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _load_model():
    # hf_hub_download() hits huggingface.co's API unconditionally (no local
    # cache check of its own) unless HF_HUB_OFFLINE is set — force it once
    # this repo is confirmed cached, so a transient HF Hub connectivity blip
    # can't crash startup for a checkpoint that's already on disk.
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


def _synthesise(tts, text: str, ref_audio: str, ref_text: str):
    """Returns a (sample_rate, mono float32 PCM ndarray) tuple."""
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
    return sample_rate, wav


# ── FastAPI app ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["tts"] = _load_model()
    yield


app = FastAPI(title="f5tts-hinglish", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    ref_audio: str
    ref_text: str
    out: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok" if "tts" in _state else "loading"}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    # async def, not plain def — keeps model calls on the event-loop thread
    # rather than FastAPI's worker thread pool (this backend's internal
    # ThreadPoolExecutor patch above already assumes sequential, single-
    # thread-at-a-time execution).
    import soundfile as sf

    tts = _state.get("tts")
    if tts is None:
        raise HTTPException(status_code=503, detail="Model still loading — retry shortly.")

    try:
        sample_rate, audio = _synthesise(tts, req.text, req.ref_audio, req.ref_text)
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
