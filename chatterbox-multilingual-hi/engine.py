#!/usr/bin/env python
"""
Chatterbox-Multilingual-hi (gagan1985/chatterbox-multilingual-hi-mlx-fp16)

Plain library module (no HTTP framework). Contract shared by every engine:
    load()         load the model once (idempotent)
    is_ready()     whether load() has finished
    run(request)   handle one request dict -> result dict; failures raise EngineError
With an "out" path the result is {"ok": True, "sample_rate": ..., "duration_s": ...} after writing
the WAV there; without one, the same dict plus "wav_bytes". vidgen calls this in-process (no HTTP);
server.py is an optional HTTP wrapper for standalone use.

Env vars: PORT (default 8002), HF_HOME, HF_TOKEN
"""
import io
import logging
import os
from dataclasses import dataclass, fields as _dc_fields
from pathlib import Path
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatterbox-multilingual-hi")

NAME = "chatterbox-multilingual-hi"
ROUTE = "/synthesize"
DEFAULT_PORT = 8002


class EngineError(Exception):
    def __init__(self, detail: str, status: int = 500):
        super().__init__(detail)
        self.detail = detail
        self.status = status

_REPO = "gagan1985/chatterbox-multilingual-hi-mlx-fp16"
_LANGUAGE_ID = "hi"
_TARGET_SR = 24_000
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


@dataclass
class SynthesizeRequest:
    text: str
    ref_audio: Optional[str] = None
    voice_id: Optional[str] = None
    out: Optional[str] = None


_state: dict = {}
_loaded = False


def load() -> None:
    global _loaded
    if _loaded:
        return
    _state["model"] = _load_model()
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
    # Model calls must stay on the thread that loaded the model (MLX's GPU
    # stream is thread-local; other backends aren't safe across threads).
    # load() and run() are always invoked on one thread — see stdio_runner / server.py.
    import soundfile as sf

    model = _state.get("model")
    if model is None:
        raise EngineError("Model still loading — retry shortly.", 503)

    try:
        sample_rate, audio = _synthesise(model, req.text, req.ref_audio, req.voice_id)
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
