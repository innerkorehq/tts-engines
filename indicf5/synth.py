#!/usr/bin/env python
"""
IndicF5 (ai4bharat/IndicF5) voice-clone synthesis CLI.

The repo installs as the f5_tts package (it's a fork of F5-TTS) but exposes
its inference API via transformers.AutoModel with trust_remote_code.

Request (stdin JSON):
    {"text": "...", "ref_audio": "/path/to/ref.wav", "ref_text": "...", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": 24000, "duration_s": F}
"""
import glob
import logging
import os
import shutil
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("indicf5")


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
    '            ref_text_len = sum(1 for c in ref_text if not c.isspace())  # vidgen-patch\n'
    '            gen_text_len = sum(1 for c in gen_text if not c.isspace())\n'
)
_DURATION_SENTINEL = "vidgen-patch"


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


def load_model():
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
        # Every call to this engine runs in a brand-new subprocess (see
        # TTSAdapter._run_engine_subprocess) — there's no in-process cache
        # that survives between calls, so without local_files_only, EVERY
        # synthesis call re-resolves the checkpoint against huggingface.co
        # over the network even though it's already cached locally. Falls
        # back to a normal (network-enabled) load only on the very first
        # run, when nothing is cached yet.
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


def synthesise(model, text: str, ref_audio: str, ref_text: str, out: str) -> tuple[int, float]:
    import numpy as np
    import soundfile as sf

    audio = model(text, ref_audio_path=ref_audio, ref_text=ref_text)
    if hasattr(audio, "dtype") and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    audio = np.array(audio, dtype=np.float32)
    sample_rate = 24000
    sf.write(out, audio, samplerate=sample_rate)
    return sample_rate, len(audio) / sample_rate


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    out = req["out"]

    with quiet_stdout():
        model = load_model()
        sample_rate, duration_s = synthesise(model, text, ref_audio, ref_text, out)
    succeed(sample_rate=sample_rate, duration_s=duration_s)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("IndicF5 synthesis failed")
        fail(str(e))
