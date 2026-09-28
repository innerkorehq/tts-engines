"""
F5-TTS synthesis service.

Loads the F5TTS_v1_Base DiT model + vocoder ONCE at startup and serves
synthesis requests over a small HTTP API (FastAPI) for the lifetime of the
process.

Endpoints:
    GET  /health       -> {"status": "ok"} once the model has finished loading.
    POST /synthesize    Request body:
                           {"text": "...", "ref_audio": "...", "ref_text": "...",
                            "remove_silence": true, "nfe_step": 32,
                            "cfg_strength": 2.0, "out": "/path/to/out.wav" (optional)}
                         If "out" is given, the server writes the WAV file to
                         that path (useful when the caller shares a
                         filesystem/volume with this service) and responds
                         with JSON: {"ok": true}
                         If "out" is omitted, the response body IS the WAV
                         audio (Content-Type: audio/wav).
                         On error: an HTTP error status with a JSON
                         {"detail": "..."} body.

Why a separate ThreadPoolExecutor patch: F5-TTS's `infer_batch_process` uses
a `ThreadPoolExecutor` internally that crashes on macOS MPS when batches run
concurrently — patched to max_workers=1 below, independent of how this
service's own HTTP layer schedules requests.

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8003)
Env vars: PORT (default 8003), HF_HOME, HF_TOKEN
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

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("f5tts")

_DEFAULT_PORT = 8003

# ── Memoise reference-clip preprocessing ─────────────────────────────────────
# `preprocess_ref_audio_text` loads the reference clip, runs pydub
# silence-detection/clipping, re-exports a trimmed temp wav, and md5-hashes
# it — purely to derive the *same* processed reference clip + text every time
# for a given voice profile. Cache it per (ref_audio, ref_text) so this work
# happens once per voice, not once per request.
_ref_preprocess_cache: dict = {}


def _cached_preprocess_ref_audio_text(ref_audio_orig, ref_text, *args, **kwargs):
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text

    key = (ref_audio_orig, ref_text)
    cached = _ref_preprocess_cache.get(key)
    if cached is None:
        cached = preprocess_ref_audio_text(ref_audio_orig, ref_text, *args, **kwargs)
        _ref_preprocess_cache[key] = cached
        logger.info("cached preprocessed reference clip for ref_audio=%s", ref_audio_orig)
    return cached


def _load_models():
    from cached_path import cached_path
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT

    logger.info("Loading F5TTS_v1_Base model + vocoder (one-time, this may take a while)…")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
    vocoder = load_vocoder(vocoder_name="vocos")
    model_obj = load_model(DiT, model_cfg, ckpt_path, mel_spec_type="vocos", vocab_file="")
    logger.info("Model + vocoder ready.")
    return model_obj, vocoder


def _synthesise_to_path(model_obj, vocoder, text: str, ref_audio: str, ref_text: str,
                         out_path: str, remove_silence: bool, nfe_step: int, cfg_strength: float) -> None:
    from f5_tts.infer.utils_infer import infer_process, remove_silence_for_generated_wav
    import soundfile as sf

    processed_ref_audio, processed_ref_text = _cached_preprocess_ref_audio_text(ref_audio, ref_text)

    final_wave, final_sample_rate, _spectrogram = infer_process(
        processed_ref_audio,
        processed_ref_text,
        text,
        model_obj,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
    )

    sf.write(out_path, final_wave, final_sample_rate)
    if remove_silence:
        remove_silence_for_generated_wav(out_path)


# ── FastAPI app ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_obj, vocoder = _load_models()
    _state["model_obj"] = model_obj
    _state["vocoder"] = vocoder
    yield


app = FastAPI(title="f5tts", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    ref_audio: str
    ref_text: str
    remove_silence: bool = True
    nfe_step: int = 32
    cfg_strength: float = 2.0
    out: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok" if "model_obj" in _state else "loading"}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    # async def, not plain def, so this stays on the event-loop thread
    # rather than FastAPI's worker thread pool — keeps model calls on one
    # consistent thread for the lifetime of the process (same rationale as
    # the ThreadPoolExecutor patch above: this backend is not safe to call
    # concurrently from arbitrary threads).
    model_obj = _state.get("model_obj")
    vocoder = _state.get("vocoder")
    if model_obj is None:
        raise HTTPException(status_code=503, detail="Model still loading — retry shortly.")

    out_path = req.out
    tmp_path: Optional[str] = None
    if not out_path:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_path = tmp.name
        out_path = tmp_path

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        _synthesise_to_path(
            model_obj, vocoder, req.text, req.ref_audio, req.ref_text,
            out_path, req.remove_silence, req.nfe_step, req.cfg_strength,
        )

        if req.out:
            return {"ok": True}

        return Response(content=Path(out_path).read_bytes(), media_type="audio/wav")
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", _DEFAULT_PORT)))
