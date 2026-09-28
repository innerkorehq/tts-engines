#!/usr/bin/env python
"""
Chatterbox-Multilingual-hi (gagan1985/chatterbox-multilingual-hi-mlx-fp16)
synthesis service.

Loads the mlx-audio Chatterbox model (T3 + voice-encoder + s3gen decoder)
ONCE at startup and serves synthesis requests over a small HTTP API
(FastAPI) for the lifetime of the process.

Endpoints:
    GET  /health       -> {"status": "ok"} once the model has finished loading.
    POST /synthesize    Request body:
                           {"text": "...", "ref_audio": "/path/to/ref.wav",
                            "voice_id": "vp_..." (optional),
                            "out": "/path/to/out.wav" (optional)}
                         "voice_id" is a stable id for the reference voice —
                         when present, the expensive ref_audio ->
                         speaker-conditioning pass is cached on disk keyed by
                         it (see the speaker-conditioning cache section
                         below), so repeat requests for the same voice skip
                         straight to generation. Omitting it falls back to
                         the always-recompute-from-ref_audio path.
                         If "out" is given, the server writes the WAV file to
                         that path (useful when the caller shares a
                         filesystem/volume with this service) and responds
                         with JSON: {"ok": true, "sample_rate": N, "duration_s": F}
                         If "out" is omitted, the response body IS the WAV
                         audio (Content-Type: audio/wav), with sample_rate/
                         duration_s in the X-Sample-Rate/X-Duration-Seconds
                         response headers.
                         On error: an HTTP error status with a JSON
                         {"detail": "..."} body.

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8002)
Env vars: PORT (default 8002), HF_HOME, HF_TOKEN
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
logger = logging.getLogger("chatterbox-multilingual-hi")

_REPO = "gagan1985/chatterbox-multilingual-hi-mlx-fp16"
_LANGUAGE_ID = "hi"
_TARGET_SR = 24_000
_DEFAULT_PORT = 8002

# ── Speaker-conditioning cache ────────────────────────────────────────────
# model.prepare_conditionals(ref_wav, ...) runs the S3 tokenizer twice,
# s3gen.embed_ref, and the voice encoder — real forward passes, not free —
# to turn a ref_audio clip into a Conditionals(t3, gen) object that
# model.generate(conds=...) can reuse directly, skipping all of that. Voice
# profiles are stable/named ("voice_id" in the request), so we cache the
# computed Conditionals per voice_id: an in-memory dict for same-process
# reuse, backed by a conds.safetensors-format file on disk (the same format
# Model.from_pretrained already knows how to read — see chatterbox.py) so
# the cache survives this process restarting. A cheap (size, mtime)
# fingerprint of ref_audio, stored in the safetensors file's own metadata,
# invalidates the cache if the underlying reference clip is ever replaced
# under the same voice_id.
_CONDS_CACHE_DIR = Path(__file__).parent / ".cache" / "speaker_conds"
_conds_mem_cache: dict[str, tuple] = {}  # voice_id -> (fingerprint, Conditionals)


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or anything that imports
    it, e.g. mlx_audio below) before HF_HUB_OFFLINE is set makes the env
    var a no-op (huggingface_hub reads it into a module-level constant at
    import time, not dynamically per call).
    """
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _load_model():
    """One-time setup: resolve offline-cache mode and load the mlx-audio
    Chatterbox model."""
    if _repo_looks_cached(_REPO):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        logger.info("Not fully cached locally yet — downloading %s…", _REPO)

    # mlx_audio also auto-downloads the shared S3TokenizerV2 weights it
    # depends on (mlx-community/S3TokenizerV2) on first run — left to its
    # own default resolution.
    if not os.environ.get("HF_TOKEN"):
        os.environ.pop("HF_TOKEN", None)

    from mlx_audio.tts.utils import load_model

    logger.info("Loading %s (one-time, this may take a while)…", _REPO)
    model = load_model(_REPO)
    logger.info("Chatterbox-Multilingual-hi ready.")
    return model


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


def _load_or_build_conds(model, voice_id: str, ref_audio: str):
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
                logger.info("Speaker-conditioning cache hit for voice_id=%r (disk).", voice_id)
                return conds
            logger.info("Speaker-conditioning cache stale for voice_id=%r — ref_audio changed.", voice_id)
        except Exception as exc:
            logger.info("Speaker-conditioning cache unreadable for voice_id=%r (%s) — recomputing.", voice_id, exc)

    logger.info("Speaker-conditioning cache miss for voice_id=%r — computing.", voice_id)
    conds = model.prepare_conditionals(ref_audio, _TARGET_SR)
    _conds_mem_cache[voice_id] = (fingerprint, conds)

    try:
        import mlx.core as mx

        _CONDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(cache_path), _conds_to_flat_dict(conds), metadata={"fingerprint": fingerprint},
        )
    except Exception as exc:
        logger.warning("Failed to write speaker-conditioning cache for voice_id=%r (%s) — continuing without it.", voice_id, exc)

    return conds


def _synthesise(model, text: str, ref_audio: Optional[str], voice_id: Optional[str]):
    """Returns a (sample_rate, mono float32 PCM ndarray) tuple."""
    import numpy as np

    kwargs: dict = {"text": text, "lang_code": _LANGUAGE_ID}
    if ref_audio and voice_id:
        kwargs["conds"] = _load_or_build_conds(model, voice_id, ref_audio)
    elif ref_audio:
        kwargs["ref_audio"] = ref_audio

    results = list(model.generate(**kwargs))
    if not results:
        raise RuntimeError("Chatterbox-Multilingual-hi produced no audio")

    audio = np.concatenate([np.array(r.audio) for r in results])
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    return sample_rate, audio


# ── FastAPI app ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = _load_model()
    yield


app = FastAPI(title="chatterbox-multilingual-hi", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    ref_audio: Optional[str] = None
    voice_id: Optional[str] = None
    out: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok" if "model" in _state else "loading"}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    # MUST be async def, not plain def: FastAPI runs plain `def` route
    # handlers in a worker thread pool, but MLX's GPU stream/command queue
    # is thread-local and was set up on the main thread during _load_model()
    # (called from the lifespan startup, which runs on the event loop
    # thread) — calling model.generate() from a different thread fails with
    # "RuntimeError: There is no Stream(gpu, 0) in current thread." An
    # `async def` handler runs directly on the event loop thread instead of
    # being offloaded, matching where the model was loaded. This blocks the
    # event loop during inference, which is fine here: one model, one
    # request processed at a time, same sequential behavior the original
    # stdin/stdout daemon had.
    import soundfile as sf

    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading — retry shortly.")

    try:
        sample_rate, audio = _synthesise(model, req.text, req.ref_audio, req.voice_id)
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
