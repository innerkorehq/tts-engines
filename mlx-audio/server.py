#!/usr/bin/env python
"""
mlx-audio synthesis service — 6 sub-model variants behind one process:
qwen3-tts, chatterbox, chatterbox-multilingual, voxtral-tts, higgs-tts,
svara-tts.

Models are loaded LAZILY (on first request for that model, since we don't
know which of the 6 sub-models will be requested first) and kept resident
for the process's lifetime.

Single-model-resident design (deliberate, not an oversight): mlx-audio
models here range from ~1-4GB+ each; keeping all 6 loaded simultaneously
would multiply memory footprint for no benefit in the common case (most
deployments repeatedly use one voice/model). So this service keeps AT MOST
ONE sub-model's weights resident at a time:
  - First request for model X: load X, remember it as current.
  - Next request, same model X: reuse the already-loaded object (warm path).
  - Next request, different model Y: drop the reference to X's model object,
    run gc.collect() (mlx-audio/MLX exposes no explicit "unload" hook we
    could find — this relies on Python GC + MLX's own lazy buffer
    reclamation), then load Y fresh and make it current.
This trades "switching models costs a reload" for "6x lower peak memory" —
acceptable since model switches are rare relative to same-model reuse.

Endpoints:
    GET  /health       -> {"status": "ok"} (does not imply any model is loaded
                          yet — models load lazily on first /synthesize call).
    POST /synthesize    Request body:
                           {"model": "qwen3-tts", "text": "...",
                            "ref_audio": "...", "ref_text": "...",
                            "language": "...", "voice": "...",
                            "out": "/path/to/out.wav" (optional)}
                         (which extra fields matter depends on "model" — see
                         README.)
                         If "out" is given, the server writes the WAV file to
                         that path and responds with JSON:
                           {"ok": true, "sample_rate": N, "duration_s": F}
                         If "out" is omitted, the response body IS the WAV
                         audio (Content-Type: audio/wav).
                         On error: an HTTP error status with a JSON
                         {"detail": "..."} body.

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8007)
Env vars: PORT (default 8007), HF_HOME, HF_TOKEN
"""
import gc
import io
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlx-audio")

_DEFAULT_PORT = 8007

# ── Model repo IDs ────────────────────────────────────────────────────────
_QWEN3_TTS_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
_CHATTERBOX_REPO = "mlx-community/chatterbox-turbo-fp16"
_CHATTERBOX_MULTILINGUAL_REPO = "mlx-community/chatterbox-multilingual-v3"
_VOXTRAL_TTS_REPO = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
_HIGGS_TTS_REPO = "bosonai/higgs-tts-3-4b"
_SVARA_TTS_REPO = "mlx-community/svara-tts-v1-4bit"

_TARGET_SR = 24_000

_VOXTRAL_VOICES = {
    "casual_male", "casual_female", "cheerful_female", "neutral_male", "neutral_female",
    "fr_male", "fr_female", "es_male", "es_female", "de_male", "de_female",
    "it_male", "it_female", "pt_male", "pt_female", "nl_male", "nl_female",
    "ar_male", "hi_male", "hi_female",
}

_SVARA_TTS_LANGUAGES = [
    "Hindi", "Bengali", "Marathi", "Telugu", "Kannada", "Tamil", "Malayalam",
    "Gujarati", "Punjabi", "Assamese", "Bhojpuri", "Magahi", "Maithili",
    "Chhattisgarhi", "Bodo", "Dogri", "Nepali", "Sanskrit", "English (Indian)",
]
_SVARA_TTS_VOICES = {
    f"{lang} ({gender})" for lang in _SVARA_TTS_LANGUAGES for gender in ("Male", "Female")
}

# mlx-audio's chatterbox model class (see mlx_audio.tts.models.chatterbox,
# distinct from chatterbox_turbo) — languages its `lang_code` param supports.
_CHATTERBOX_MULTILINGUAL_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}

_KNOWN_MODELS = {
    "qwen3-tts", "chatterbox", "chatterbox-multilingual", "voxtral-tts",
    "higgs-tts", "svara-tts",
}

# ── Single-resident-model state ──────────────────────────────────────────────
# At most one of these is non-None at any time — whichever model
# `_current_model_name` names. Everything else is None.
_current_model_name: Optional[str] = None
_current_model_obj = None


def _repo_looks_cached(repo: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or anything that imports it,
    e.g. mlx_audio below) before HF_HUB_OFFLINE is set makes the env var a
    no-op (huggingface_hub reads it into a module-level constant at import
    time, not dynamically per call).
    """
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _load_model_local_first(repo: str):
    if _repo_looks_cached(repo):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        logger.info("Not fully cached locally yet — downloading %s…", repo)

    from mlx_audio.tts.utils import load_model

    return load_model(repo)


def _unload_current_model() -> None:
    """
    Drop whatever model is currently resident so its weights can be
    reclaimed before loading a different one.

    Neither `mlx_audio.tts.utils.load_model()`'s return objects nor the MLX
    framework itself expose an explicit "unload"/"free" call as far as this
    code found — MLX arrays are lazily-evaluated and reference-counted, so
    dropping the last Python reference plus a `gc.collect()` is the
    documented way to let their backing memory be released.
    """
    global _current_model_obj, _current_model_name
    if _current_model_obj is not None:
        logger.info("Unloading current model (%s) before switching…", _current_model_name)
    _current_model_obj = None
    _current_model_name = None
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass


def _ensure_model_loaded(model_name: str):
    """
    Return the resident model object for `model_name`, loading it (and
    evicting whatever was previously resident) if it isn't already current.
    """
    global _current_model_obj, _current_model_name

    if model_name == _current_model_name and _current_model_obj is not None:
        return _current_model_obj  # warm reuse — no reload needed

    if _current_model_name is not None:
        _unload_current_model()

    repo = {
        "qwen3-tts": _QWEN3_TTS_REPO,
        "chatterbox": _CHATTERBOX_REPO,
        "chatterbox-multilingual": _CHATTERBOX_MULTILINGUAL_REPO,
        "voxtral-tts": _VOXTRAL_TTS_REPO,
        "higgs-tts": _HIGGS_TTS_REPO,
        "svara-tts": _SVARA_TTS_REPO,
    }[model_name]

    logger.info("Loading %s (%s)…", model_name, repo)
    model_obj = _load_model_local_first(repo)
    logger.info("%s ready.", model_name)

    _current_model_obj = model_obj
    _current_model_name = model_name
    return model_obj


# ── Per-model inference ──────────────────────────────────────────────────────

def _synthesise_qwen3_tts(model, text: str, ref_audio: str, ref_text: str, language: str):
    import numpy as np

    results = list(model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, language=language or "English"))
    if not results:
        raise RuntimeError("Qwen3-TTS produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    return getattr(results[0], "sample_rate", _TARGET_SR), audio


def _synthesise_chatterbox(model, text: str, ref_audio: str):
    import numpy as np
    import soundfile as sf

    kwargs = {"text": text}
    if ref_audio:
        # chatterbox-turbo hard-asserts on this (AssertionError, not a
        # friendly message) — check it ourselves so the failure is clear.
        info = sf.info(ref_audio)
        if info.duration <= 5.0:
            raise ValueError(
                f"Chatterbox requires a reference clip longer than 5 seconds "
                f"(got {info.duration:.1f}s: {ref_audio})"
            )
        kwargs["ref_audio"] = ref_audio
    results = list(model.generate(**kwargs))
    if not results:
        raise RuntimeError("Chatterbox produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    return getattr(results[0], "sample_rate", _TARGET_SR), audio


def _synthesise_chatterbox_multilingual(model, text: str, ref_audio: str, language: str):
    import numpy as np

    lang_code = (language or "en").lower()
    if lang_code not in _CHATTERBOX_MULTILINGUAL_LANGUAGES:
        raise ValueError(
            f"Unknown Chatterbox-Multilingual language {language!r}; "
            f"expected one of {sorted(_CHATTERBOX_MULTILINGUAL_LANGUAGES)}"
        )

    kwargs: dict = {"text": text, "lang_code": lang_code}
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
    results = list(model.generate(**kwargs))
    if not results:
        raise RuntimeError("Chatterbox-Multilingual produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    return getattr(results[0], "sample_rate", _TARGET_SR), audio


def _synthesise_voxtral_tts(model, text: str, voice: str):
    import numpy as np

    if voice not in _VOXTRAL_VOICES:
        raise ValueError(f"Unknown Voxtral-TTS voice {voice!r}; expected one of {sorted(_VOXTRAL_VOICES)}")
    if voice.startswith("hi_") and text.isascii():
        logger.warning(
            "voice=%s expects Devanagari-script Hindi, but text is pure ASCII "
            "(romanised?) — verified this produces garbled/repetition-looping "
            "audio, not just degraded quality. Pass genuine Devanagari text.",
            voice,
        )

    results = list(model.generate(text=text, voice=voice))
    if not results:
        raise RuntimeError("Voxtral-TTS produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    return getattr(results[0], "sample_rate", _TARGET_SR), audio


def _synthesise_svara_tts(model, text: str, voice: str):
    import numpy as np

    if voice not in _SVARA_TTS_VOICES:
        raise ValueError(f"Unknown Svara-TTS voice {voice!r}; expected one of {sorted(_SVARA_TTS_VOICES)}")

    results = list(model.generate(
        text=text, voice=voice,
        temperature=0.75, top_p=0.9, top_k=40, repetition_penalty=1.1, max_tokens=1200,
    ))
    if not results:
        raise RuntimeError("Svara-TTS produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    sample_rate = getattr(results[0], "sample_rate", None) or getattr(model, "sample_rate", _TARGET_SR)
    return sample_rate, audio


def _synthesise_higgs_tts(model, text: str, ref_audio: str, ref_text: str):
    import numpy as np

    # lang_code is not passed — higgs_audio_v3.generate() deletes **kwargs and
    # auto-detects language from the text content.
    gen_kwargs: dict = {"text": text, "verbose": False}
    if ref_audio:
        gen_kwargs["ref_audio"] = ref_audio
    if ref_text:
        gen_kwargs["ref_text"] = ref_text
    results = list(model.generate(**gen_kwargs))
    if not results:
        raise RuntimeError("Higgs-TTS produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    return getattr(results[0], "sample_rate", _TARGET_SR), audio


def _synthesise(model_name: str, text: str, req: "SynthesizeRequest"):
    """Returns a (sample_rate, mono float32 PCM ndarray) tuple."""
    model = _ensure_model_loaded(model_name)

    if model_name == "qwen3-tts":
        return _synthesise_qwen3_tts(model, text, req.ref_audio or "", req.ref_text or "", req.language or "English")
    elif model_name == "chatterbox":
        return _synthesise_chatterbox(model, text, req.ref_audio or "")
    elif model_name == "chatterbox-multilingual":
        return _synthesise_chatterbox_multilingual(model, text, req.ref_audio or "", req.language or "en")
    elif model_name == "voxtral-tts":
        return _synthesise_voxtral_tts(model, text, req.voice or "neutral_female")
    elif model_name == "higgs-tts":
        return _synthesise_higgs_tts(model, text, req.ref_audio or "", req.ref_text or "")
    elif model_name == "svara-tts":
        return _synthesise_svara_tts(model, text, req.voice or "Hindi (Female)")
    else:  # unreachable — guarded by _KNOWN_MODELS check in the route
        raise AssertionError(f"unhandled model {model_name!r}")


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="mlx-audio")


class SynthesizeRequest(BaseModel):
    model: str
    text: str
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    language: Optional[str] = None
    voice: Optional[str] = None
    out: Optional[str] = None


@app.get("/health")
def health():
    # Does not imply any model is loaded yet — models load lazily on first
    # /synthesize call for that model (see module docstring).
    return {"status": "ok", "resident_model": _current_model_name}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    # async def, not plain def: MLX's GPU stream/command queue is
    # thread-local, and FastAPI runs plain `def` route handlers in a worker
    # thread pool — calling into an MLX model from a different thread than
    # it was loaded on fails with "RuntimeError: There is no Stream(gpu, 0)
    # in current thread." An `async def` handler stays on the event-loop
    # thread instead, matching where models get loaded on first use.
    import soundfile as sf

    if req.model not in _KNOWN_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model: {req.model!r} (expected one of {sorted(_KNOWN_MODELS)})",
        )
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is empty")

    try:
        sample_rate, audio = _synthesise(req.model, req.text, req)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
