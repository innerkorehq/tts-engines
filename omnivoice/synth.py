#!/usr/bin/env python
"""
k2-fsa/OmniVoice TTS CLI — official PyTorch `omnivoice` package, not the
mlx-audio port. Switched from the mlx-audio/MLX backend because the
mlx-community/OmniVoice-bf16 checkpoint was missing weights mlx-audio's loader
expected (see git history for that workaround). The official package loads
the same k2-fsa/OmniVoice checkpoint directly, with no such loading bug.

Massively multilingual (600+ languages) zero-shot TTS, diffusion-language-model
architecture. Runs on Apple Silicon via PyTorch's MPS backend.
  https://huggingface.co/k2-fsa/OmniVoice
  https://github.com/k2-fsa/OmniVoice

Unlike the old mlx-audio path, this model does NOT take a "language" hint —
it infers language/script directly from the input text. `language_code` in
the voice profile is still used upstream (pipeline_service.py) to pick the
number-normalization language, but is not passed to this script.

Voice modes (matching the model's three native modes — see
https://github.com/k2-fsa/OmniVoice#python-api):
  - Voice cloning: ref_audio (+ optional ref_text — auto-transcribed via
    Whisper if omitted)
  - Auto voice: neither ref_audio nor ref_text given — model picks a voice.

KNOWN MODEL INSTABILITY ON SHORT TEXT — mitigated below, see
_is_degenerate()/synthesise(): same instability observed on the old mlx-audio
backend also reproduces here, so it's a property of OmniVoice's generation
itself, not the inference backend. Single words/short phrases (e.g. "Hello",
"GitHub") occasionally produce a degenerate result — a loud, sustained,
pause-free buzz filling the whole clip — verified via Whisper transcription
returning gibberish/silence for these. Detected via a cheap RMS/peak
heuristic and retried with a fresh generation, up to a few attempts.

MPS OUT-OF-MEMORY — mitigated below, see _generate()/_get_cpu_model():
the MPS shared-memory pool is shared system-wide across every process using
Metal (e.g. a concurrent Remotion/Chromium render), so this subprocess can
hit "MPS backend out of memory" even though IT alone never allocates much —
seen in production with "other allocations: 16.48 GiB" dwarfing our own
~3.6 GiB. Loading in float16 (instead of float32) halves our footprint, and
on an actual OOM we fall back to a CPU copy of the model for that call
(slower, but the job succeeds instead of failing outright).

Request (stdin JSON):
    {"text": "...", "out": "/abs/out.wav",
     "ref_audio": "/abs/ref.wav" (optional), "ref_text": "..." (optional)}

Response (stdout JSON): {"ok": true, "sample_rate": 24000, "duration_s": F}
                         or {"ok": false, "error": "..."}
"""
import logging
import sys
from pathlib import Path

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("omnivoice")

_MODEL_ID = "k2-fsa/OmniVoice"

# Quality gate for the short-text-degeneration issue (see module docstring).
# Thresholds derived from real samples: healthy short clips ("API", "URL")
# measured rms/peak ~0.20; degenerate clips ("Hello", "GitHub" — confirmed
# gibberish/silent via Whisper) measured rms/peak ~0.41-0.49 (continuous loud
# buzz, no pauses, unlike real speech which has bursts/pauses pulling rms
# well below peak).
_MIN_PEAK = 0.01
_MAX_RMS_TO_PEAK_RATIO = 0.35
_MAX_GENERATION_ATTEMPTS = 4

_model = None
_cpu_model = None
_force_cpu = False  # sticky once an MPS OOM is hit — MPS is contended for the rest of this run


def _get_model():
    global _model
    if _model is not None:
        return _model

    import torch
    from omnivoice import OmniVoice

    device_map = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device_map == "mps" else torch.float32
    logger.info("Loading %s (device_map=%s, dtype=%s)…", _MODEL_ID, device_map, dtype)
    _model = _from_pretrained_local_first(OmniVoice, device_map=device_map, dtype=dtype)
    logger.info("OmniVoice ready.")
    return _model


def _get_cpu_model():
    global _cpu_model
    if _cpu_model is not None:
        return _cpu_model

    import torch
    from omnivoice import OmniVoice

    logger.info("Loading %s (device_map=cpu fallback)…", _MODEL_ID)
    _cpu_model = _from_pretrained_local_first(OmniVoice, device_map="cpu", dtype=torch.float32)
    logger.info("OmniVoice (CPU fallback) ready.")
    return _cpu_model


def _from_pretrained_local_first(model_cls, **kwargs):
    """
    Every call to this engine runs in a brand-new subprocess (see
    TTSAdapter._run_engine_subprocess) — there's no in-process cache that
    survives between calls, so without local_files_only, EVERY synthesis
    call re-resolves every config/tokenizer/checkpoint file against
    huggingface.co over the network, even though the weights are already
    cached locally. Falls back to a normal (network-enabled) load only on
    the very first run, when nothing is cached yet.
    """
    try:
        return model_cls.from_pretrained(_MODEL_ID, local_files_only=True, **kwargs)
    except Exception:
        logger.info("Not fully cached locally yet — downloading %s…", _MODEL_ID)
        return model_cls.from_pretrained(_MODEL_ID, **kwargs)


def _is_mps_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _generate(kwargs: dict):
    """Run model.generate(), falling back to a CPU copy of the model on an
    MPS out-of-memory error (see module docstring) instead of failing."""
    global _force_cpu

    if not _force_cpu:
        try:
            model = _get_model()
            return model.generate(**kwargs)[0]
        except RuntimeError as e:
            if not _is_mps_oom(e):
                raise
            import torch
            logger.warning("MPS out of memory (%s) — falling back to CPU for this and remaining calls", e)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            _force_cpu = True

    return _get_cpu_model().generate(**kwargs)[0]


def _is_degenerate(audio) -> bool:
    import numpy as np

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < _MIN_PEAK:
        return True
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return (rms / peak) > _MAX_RMS_TO_PEAK_RATIO


def synthesise(text: str, out_path: str, ref_audio: str = "", ref_text: str = "") -> tuple[int, float]:
    import soundfile as sf

    sample_rate = 24000

    kwargs: dict = {"text": text}
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
        if ref_text:
            kwargs["ref_text"] = ref_text
        else:
            logger.info("ref_audio given without ref_text — model will auto-transcribe via Whisper")

    audio = _generate(kwargs)
    attempt = 1
    while _is_degenerate(audio) and attempt < _MAX_GENERATION_ATTEMPTS:
        attempt += 1
        logger.warning(
            "Generation %d produced degenerate audio (known short-text instability — "
            "see module docstring); retrying (attempt %d/%d)…",
            attempt - 1, attempt, _MAX_GENERATION_ATTEMPTS,
        )
        audio = _generate(kwargs)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    out_path = req.get("out")
    ref_audio = req.get("ref_audio", "")
    ref_text = req.get("ref_text", "")

    if not text.strip():
        fail("text is empty")
        return
    if not out_path:
        fail("out path is required")
        return

    try:
        with quiet_stdout():
            sample_rate, duration_s = synthesise(text, out_path, ref_audio, ref_text)
    except Exception as e:
        logger.exception("omnivoice synthesis failed")
        fail(str(e))
        return

    succeed(sample_rate=sample_rate, duration_s=duration_s)


if __name__ == "__main__":
    main()
