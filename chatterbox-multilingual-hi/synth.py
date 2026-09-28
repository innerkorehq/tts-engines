#!/usr/bin/env python
"""
Chatterbox-Multilingual-hi (gagan1985/chatterbox-multilingual-hi-mlx-fp16)
voice-clone synthesis CLI — native MLX (Apple Silicon) via mlx-audio.

gagan1985/chatterbox-multilingual-hi-mlx-fp16 repackages ResembleAI's Hindi
Chatterbox finetune (t3_hi.safetensors, from
ResembleAI/Chatterbox-Multilingual-hi) alongside the shared base model's
voice-encoder/speech-decoder weights (ve.safetensors/s3gen.safetensors, from
ResembleAI/chatterbox — only T3 was fine-tuned for Hindi) for mlx-audio's own
Chatterbox model classes, converted with mlx-audio's own sanitize() — the
same code path used for the official mlx-community Chatterbox conversions.
Previously this engine ran the PyTorch chatterbox-tts package directly
(loading the base model then manually swapping the T3 state_dict for the
Hindi finetune); this repo does that same T3-swap ahead of time, packaged
for mlx-audio to load in one call.

Request (stdin JSON):
    {"text": "...", "ref_audio": "/path/to/ref.wav", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": N, "duration_s": F}
"""
import logging
import os
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("chatterbox-multilingual-hi")

_REPO = "gagan1985/chatterbox-multilingual-hi-mlx-fp16"
_LANGUAGE_ID = "hi"
_TARGET_SR = 24_000

_model = None


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or anything that imports
    it, e.g. mlx_audio below) before HF_HUB_OFFLINE is set makes the env
    var a no-op (huggingface_hub reads it into a module-level constant at
    import time, not dynamically per call).
    """
    from pathlib import Path

    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _get_model():
    global _model
    if _model is not None:
        return _model

    # Every call to this engine runs in a brand-new subprocess (see
    # TTSAdapter._run_engine_subprocess) — there's no in-process cache that
    # survives between calls, so without forcing offline mode, EVERY
    # synthesis call re-resolves every checkpoint file against
    # huggingface.co over the network, even though it's already cached
    # locally. mlx_audio.tts.utils.load_model() doesn't expose a
    # local_files_only-style kwarg, so HF_HUB_OFFLINE forces the same
    # behavior transparently — but it MUST be set before importing
    # mlx_audio (it's read into a module-level constant at import time, not
    # rechecked per call), so we check the cache directory directly first
    # rather than importing mlx_audio in a try/except.
    if _repo_looks_cached(_REPO):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        logger.info("Not fully cached locally yet — downloading %s…", _REPO)

    # mlx_audio also auto-downloads the shared S3TokenizerV2 weights it
    # depends on (mlx-community/S3TokenizerV2) on first run — left to its
    # own default resolution, same as the mlx-audio engine's other models.
    if not os.environ.get("HF_TOKEN"):
        os.environ.pop("HF_TOKEN", None)

    from mlx_audio.tts.utils import load_model

    logger.info("Loading %s…", _REPO)
    _model = load_model(_REPO)
    logger.info("Chatterbox-Multilingual-hi ready.")
    return _model


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    out = req["out"]

    import numpy as np
    import soundfile as sf

    with quiet_stdout():
        model = _get_model()
        kwargs: dict = {"text": text, "lang_code": _LANGUAGE_ID}
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
        results = list(model.generate(**kwargs))
        if not results:
            raise RuntimeError("Chatterbox-Multilingual-hi produced no audio")
        audio = np.concatenate([np.array(r.audio) for r in results])
        sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
        sf.write(out, audio, sample_rate)
    succeed(sample_rate=sample_rate, duration_s=len(audio) / sample_rate)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Chatterbox-Multilingual-hi synthesis failed")
        fail(str(e))
