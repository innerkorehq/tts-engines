"""
F5-TTS-Hinglish synthesis service.

Plain library module (no HTTP framework). Contract shared by every engine:
    load()         load the model once (idempotent)
    is_ready()     whether load() has finished
    run(request)   handle one request dict -> result dict; failures raise EngineError
With an "out" path the result is {"ok": True, "sample_rate": ..., "duration_s": ...} after writing
the WAV there; without one, the same dict plus "wav_bytes". vidgen calls this in-process (no HTTP);
server.py is an optional HTTP wrapper for standalone use.

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
from dataclasses import dataclass, fields as _dc_fields
from pathlib import Path
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("f5tts-hinglish")

NAME = "f5tts-hinglish"
ROUTE = "/synthesize"
DEFAULT_PORT = 8004


class EngineError(Exception):
    def __init__(self, detail: str, status: int = 500):
        super().__init__(detail)
        self.detail = detail
        self.status = status

_HF_REPO = "rajputsw/F5-TTS-Hinglish"
_MODEL_CONFIG = "F5TTS_Small"  # 768-dim/18-layer/12-head DiT, matches this checkpoint
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


def _load_model():
    _use_soundfile_for_torchaudio_load()
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
    _state["tts"] = _load_model()
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

    tts = _state.get("tts")
    if tts is None:
        raise EngineError("Model still loading — retry shortly.", 503)

    try:
        sample_rate, audio = _synthesise(tts, req.text, req.ref_audio, req.ref_text)
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
