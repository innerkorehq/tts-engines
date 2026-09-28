"""
IndicF5 (ai4bharat/IndicF5) synthesis service.

Loads the IndicF5 model (which bundles its own vocoder, loaded via
`transformers.AutoModel(..., trust_remote_code=True)`) ONCE at startup and
serves synthesis requests over a small HTTP API (FastAPI) for the lifetime
of the process.

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

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8011)
Env vars: PORT (default 8011), HF_TOKEN (ai4bharat/IndicF5 is gated)
"""
import glob
import io
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indicf5")

_DEFAULT_PORT = 8011


def _patch_indicf5_for_mps() -> None:
    """
    IndicF5 ships its model code via HF `trust_remote_code` (downloaded to
    ~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/<rev>/model.py).
    That file hardcodes:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    — i.e. it never considers Apple Silicon's MPS backend, so on macOS it
    always falls back to CPU (slow). Patch the cached file in-place to add an
    MPS branch. Idempotent (checks a marker before writing) and self-heals if
    HF re-downloads a fresh copy of the file.
    """
    try:
        import torch
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return  # nothing to do on non-Apple-Silicon machines
    except Exception:
        return

    cache_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5")
    marker = "torch.backends.mps.is_available()"
    old = 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
    new = (
        'device = torch.device(\n'
        '            "cuda" if torch.cuda.is_available()\n'
        '            else "mps" if torch.backends.mps.is_available()\n'
        '            else "cpu"\n'
        '        )'
    )

    for path in glob.glob(os.path.join(cache_root, "*", "model.py")):
        try:
            src = open(path, "r", encoding="utf-8").read()
            if marker in src:
                continue  # already patched
            if old not in src:
                logger.warning("IndicF5 model.py at %s has unexpected device-selection code — skipping MPS patch.", path)
                continue
            patched = src.replace(old, new, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(patched)
            shutil.rmtree(os.path.join(os.path.dirname(path), "__pycache__"), ignore_errors=True)
            logger.info("Patched IndicF5 model.py for Apple Silicon MPS support: %s", path)
        except Exception as e:
            logger.warning("Could not patch IndicF5 model.py at %s for MPS: %s", path, e)


# f5_tts.infer.utils_infer.infer_process() (vendored into IndicF5's model.py
# via `from f5_tts.infer.utils_infer import ...`) computes the output audio
# canvas size from UTF-8 *byte* counts of the reference/generation texts:
#
#   ref_text_len = len(ref_text.encode("utf-8"))
#   gen_text_len = len(gen_text.encode("utf-8"))
#
# Devanagari encodes as ~3 bytes/char vs ~1 byte/char for ASCII. For
# Roman-script input this under-allocates canvas relative to a Devanagari
# generation text, truncating the synthesised audio. Counting non-whitespace
# *characters* instead fixes the ref:gen ratio regardless of script. This
# mirrors the "Mode A" duration fix from https://github.com/harrrshall/hinglish-tts.
_DURATION_ORIG_BLOCK = (
    '            # Calculate duration\n'
    '            ref_text_len = len(ref_text.encode("utf-8"))\n'
    '            gen_text_len = len(gen_text.encode("utf-8"))\n'
)
_DURATION_PATCHED_BLOCK = (
    '            # PATCHED — character-count proportional duration (Mode A fix)\n'
    '            ref_text_len = sum(1 for c in ref_text if not c.isspace())  # indicf5-patch\n'
    '            gen_text_len = sum(1 for c in gen_text if not c.isspace())\n'
)
_DURATION_SENTINEL = "indicf5-patch"


def _patch_indicf5_duration_canvas() -> None:
    """Patch the installed f5_tts package's utils_infer.py in place — see block comment above."""
    try:
        import f5_tts.infer.utils_infer as _utils_infer
    except Exception:
        logger.warning("IndicF5 duration patch: f5_tts not importable — skipping.")
        return

    path = _utils_infer.__file__
    try:
        src = open(path, "r", encoding="utf-8").read()
        if _DURATION_SENTINEL in src:
            return  # already patched
        if _DURATION_ORIG_BLOCK not in src:
            logger.warning("IndicF5 duration patch: utils_infer.py at %s has unexpected code — skipping.", path)
            return
        patched = src.replace(_DURATION_ORIG_BLOCK, _DURATION_PATCHED_BLOCK, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        shutil.rmtree(os.path.join(os.path.dirname(path), "__pycache__"), ignore_errors=True)
        logger.info("Patched f5_tts utils_infer.py for character-count duration: %s", path)
    except Exception as e:
        logger.warning("Could not patch f5_tts utils_infer.py at %s for duration: %s", path, e)


def _load_model():
    """Load ai4bharat/IndicF5 via transformers.AutoModel(trust_remote_code=True)."""
    import torch as _torch
    import transformers.modeling_utils as _mu
    from transformers import AutoModel

    _patch_indicf5_for_mps()
    _patch_indicf5_duration_canvas()

    # transformers ALWAYS constructs `cls(config, ...)` inside a meta-device
    # context (regardless of `low_cpu_mem_usage`). IndicF5's custom __init__
    # loads its own vocoder checkpoint and calls plain `.to(device)` on it —
    # that blows up with "Cannot copy out of meta tensor; no data!" because
    # the vocoder submodules were created on the meta device. Temporarily
    # strip the meta-device context so the whole model (including the
    # vocoder loaded inside __init__) is built with real, materialised
    # tensors.
    _PTM = _mu.PreTrainedModel
    _orig_get_init_context = _PTM.get_init_context.__func__

    @classmethod
    def _get_init_context_no_meta(cls_, dtype, is_quantized, _is_ds_init_called):
        contexts = _orig_get_init_context(cls_, dtype, is_quantized, _is_ds_init_called)
        return [c for c in contexts if c != _torch.device("meta")]

    # IndicF5's custom INF5Model.__init__ calls `super().__init__(config)` but
    # never calls `self.post_init()` (its code predates the transformers
    # version that requires it). Without it, `all_tied_weights_keys` /
    # `_tp_plan` / `_no_split_modules` etc. are never set, and
    # `_finalize_model_loading` blows up. Run `post_init()` retroactively —
    # it only computes this metadata from the already-built module tree.
    _orig_finalize = _PTM._finalize_model_loading

    @staticmethod
    def _finalize_model_loading_compat(model, load_config, loading_info):
        if not hasattr(model, "all_tied_weights_keys"):
            model.post_init()
        return _orig_finalize(model, load_config, loading_info)

    _PTM.get_init_context = _get_init_context_no_meta
    _PTM._finalize_model_loading = _finalize_model_loading_compat
    try:
        logger.info("Loading IndicF5 model from ai4bharat/IndicF5…")
        try:
            model = AutoModel.from_pretrained(
                "ai4bharat/IndicF5", trust_remote_code=True, low_cpu_mem_usage=False, local_files_only=True,
            )
        except Exception:
            logger.info("Not fully cached locally yet — downloading ai4bharat/IndicF5…")
            model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, low_cpu_mem_usage=False)
    finally:
        _PTM.get_init_context = classmethod(_orig_get_init_context)
        _PTM._finalize_model_loading = staticmethod(_orig_finalize)
    logger.info("IndicF5 model ready.")
    return model


def _synthesise(model, text: str, ref_audio: str, ref_text: str):
    """Returns a (sample_rate, mono float32 PCM ndarray) tuple."""
    import numpy as np

    audio = model(text, ref_audio_path=ref_audio, ref_text=ref_text)
    if hasattr(audio, "dtype") and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    audio = np.array(audio, dtype=np.float32)
    return 24000, audio


# ── FastAPI app ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading IndicF5 model (one-time, this may take a while)…")
    _state["model"] = _load_model()
    logger.info("Model ready.")
    yield


app = FastAPI(title="indicf5", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    ref_audio: str
    ref_text: str
    out: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok" if "model" in _state else "loading"}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    import soundfile as sf

    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading — retry shortly.")

    try:
        sample_rate, audio = _synthesise(model, req.text, req.ref_audio, req.ref_text)
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    duration_s = len(audio) / sample_rate

    if req.out:
        Path(req.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(req.out, audio, samplerate=sample_rate)
        return {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=sample_rate, format="WAV")
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
