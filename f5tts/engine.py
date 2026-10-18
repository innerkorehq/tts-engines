"""
F5-TTS synthesis service.

Plain library module (no HTTP framework). Contract shared by every engine:
    load()         load the model once (idempotent)
    is_ready()     whether load() has finished
    run(request)   handle one request dict -> result dict; failures raise EngineError
With an "out" path the result is {"ok": True, "sample_rate": ..., "duration_s": ...} after writing
the WAV there; without one, the same dict plus "wav_bytes". vidgen calls this in-process (no HTTP);
server.py is an optional HTTP wrapper for standalone use.

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
from dataclasses import dataclass, fields as _dc_fields
from pathlib import Path
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("f5tts")

NAME = "f5tts"
ROUTE = "/synthesize"
DEFAULT_PORT = 8003


class EngineError(Exception):
    def __init__(self, detail: str, status: int = 500):
        super().__init__(detail)
        self.detail = detail
        self.status = status

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


def _use_soundfile_for_torchaudio_load() -> None:
    """torchaudio >= 2.9 routes load() through torchcodec, whose shared libraries
    need ffmpeg <= 8 and fail to load against newer system ffmpeg builds. Decode
    with soundfile instead (a dependency already), same (tensor, sample_rate)."""
    import soundfile as sf
    import torch
    import torchaudio

    def load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None,
             buffer_size=4096, backend=None):
        data, sample_rate = sf.read(uri, start=frame_offset, frames=num_frames, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data)
        return (tensor.T.contiguous() if channels_first else tensor), sample_rate

    torchaudio.load = load


def _load_models():
    _use_soundfile_for_torchaudio_load()
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


@dataclass
class SynthesizeRequest:
    text: str
    ref_audio: str
    ref_text: str
    remove_silence: bool = True
    nfe_step: int = 32
    cfg_strength: float = 2.0
    out: Optional[str] = None


_state: dict = {}
_loaded = False


def load() -> None:
    global _loaded
    if _loaded:
        return
    model_obj, vocoder = _load_models()
    _state["model_obj"] = model_obj
    _state["vocoder"] = vocoder
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
    model_obj = _state.get("model_obj")
    vocoder = _state.get("vocoder")
    if model_obj is None:
        raise EngineError("Model still loading — retry shortly.", 503)

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

        return {"ok": True, "wav_bytes": Path(out_path).read_bytes()}
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise EngineError(f"{type(exc).__name__}: {exc}", 500) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
