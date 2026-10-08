#!/usr/bin/env python
"""
hinglish-tts synthesis service.

Plain library module (no HTTP framework). Contract shared by every engine:
    load()         load the model once (idempotent)
    is_ready()     whether load() has finished
    run(request)   handle one request dict -> result dict; failures raise EngineError
With an "out" path the result is {"ok": True, "sample_rate": ..., "duration_s": ...} after writing
the WAV there; without one, the same dict plus "wav_bytes". vidgen calls this in-process (no HTTP);
server.py is an optional HTTP wrapper for standalone use.

Env vars: PORT (default 8005), HF_HOME, HF_TOKEN
"""
import io
import logging
import os
from dataclasses import dataclass, fields as _dc_fields
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hinglish-tts")

NAME = "hinglish-tts"
ROUTE = "/synthesize"
DEFAULT_PORT = 8005


class EngineError(Exception):
    def __init__(self, detail: str, status: int = 500):
        super().__init__(detail)
        self.detail = detail
        self.status = status

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

_SAMPLE_RATE = 24_000


@dataclass
class SynthesizeRequest:
    text: str
    ref_audio: str
    ref_text: str
    out: Optional[str] = None


_state: dict = {}
_loaded = False


def load() -> None:
    global _loaded
    if _loaded:
        return
    logger.info("Loading IndicF5 via inference.load_model()…")
    _state["model"] = load_model()
    logger.info("IndicF5 ready.")
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

    model = _state.get("model")
    if model is None:
        raise EngineError("Model still loading — retry shortly.", 503)

    try:
        audio = synthesize(model, req.text, ref_audio_path=req.ref_audio, ref_text=req.ref_text)
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise EngineError(f"{type(exc).__name__}: {exc}", 500) from exc

    duration_s = len(audio) / _SAMPLE_RATE

    if req.out:
        Path(req.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(req.out, audio, samplerate=_SAMPLE_RATE)
        return {"ok": True, "sample_rate": _SAMPLE_RATE, "duration_s": duration_s}

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=_SAMPLE_RATE, format="WAV")
    return {"ok": True, "sample_rate": _SAMPLE_RATE, "duration_s": float(duration_s), "wav_bytes": buf.getvalue()}
