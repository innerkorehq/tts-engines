#!/usr/bin/env python
"""
Consolidated mlx-audio TTS CLI — Qwen3-TTS, Chatterbox, Voxtral-TTS, all
running natively on Apple Silicon via mlx-audio (https://github.com/
Blaizzy/mlx-audio), one shared venv instead of one per model.

Deliberately NOT consolidated here:
  - Kokoro: mlx-audio 0.4.4's Kokoro port has a verified, reproducible,
    currently-open upstream regression (introduced between 0.4.1 and 0.4.4)
    — certain ordinary short inputs (e.g. literally "Hello world") crash
    deterministically with `ValueError: [broadcast_shapes] Shapes (1,N,1)
    and (1,N+300,9) cannot be broadcast` inside the SineGen vocoder
    component. See github.com/Blaizzy/mlx-audio issues #784/#786 and PRs
    #785/#788 — both fix attempts still unmerged at time of writing. Kokoro
    is the default, most heavily-used engine in this app; not worth the
    risk. Stays on the existing tts-engines/kokoro/ engine (plain `kokoro`
    PyPI package, proven stable).
  - OmniVoice: mlx-audio's port (mlx-community/OmniVoice-bf16) has a verified
    upstream bug — the checkpoint is missing weights its own config declares
    (semantic_model.*), so mlx-audio's loader silently sets
    model.audio_tokenizer = None and generation produces degenerate audio
    with no error. Re-tested against the current mlx-audio release (0.4.4)
    and still reproduces. OmniVoice stays on the official k2-fsa PyTorch
    package (tts-engines/omnivoice/).
  - chatterbox-hinglish, f5tts-hinglish, indicf5-hinglish, veena-hinglish:
    fine-tuned Hinglish checkpoints on architectures mlx-audio either
    doesn't implement at all (F5-TTS, IndicF5, Veena/SNAC) or where no
    pre-converted MLX checkpoint exists for that specific fine-tune
    (chatterbox-turbo-hinglish). Kept on their existing isolated engines.

Request (stdin JSON):
    {"model": "qwen3-tts" | "chatterbox" | "voxtral-tts",
     "text": "...", "out": "/abs/out.wav", ...model-specific fields}

  qwen3-tts:   {"ref_audio": "/abs/ref.wav", "ref_text": "...",
                "language": "English"}  (zero-shot voice cloning)
  chatterbox:  {"ref_audio": "/abs/ref.wav"}  (zero-shot voice cloning,
                no ref_text needed — Resemble's voice encoder doesn't
                require a matching transcript. Reference clip MUST be
                longer than 5 seconds — chatterbox-turbo hard-asserts on
                this.)
  voxtral-tts: {"voice": "hi_male"}  (preset voices only, no cloning —
                see _VOXTRAL_VOICES). For hi_male/hi_female: feed genuine
                Devanagari-script Hindi, NOT romanised Hinglish — tested
                romanised input ("Aaj ki badi khabar...") and got a garbled,
                repetition-looping result; the same sentence in Devanagari
                produced clean, correct Hindi speech. This is NOT a Hinglish
                (code-switched) engine — that's still the dedicated
                *-hinglish engines kept outside this consolidation.

Response (stdout JSON): {"ok": true, "sample_rate": N, "duration_s": F}
                         or {"ok": false, "error": "..."}
"""
import logging
import sys
from pathlib import Path

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mlx-audio")

_QWEN3_TTS_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
_CHATTERBOX_REPO = "mlx-community/chatterbox-turbo-fp16"
_VOXTRAL_TTS_REPO = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
_HIGGS_TTS_REPO = "bosonai/higgs-tts-3-4b"
_higgs_tts_model = None

_TARGET_SR = 24_000

_qwen3_tts_model = None
_chatterbox_model = None
_voxtral_tts_model = None


def _repo_looks_cached(repo: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or anything that imports
    it, e.g. mlx_audio below) before HF_HUB_OFFLINE is set makes the env
    var a no-op (huggingface_hub reads it into a module-level constant at
    import time, not dynamically per call).
    """
    import os
    from pathlib import Path

    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _load_model_local_first(repo: str):
    """
    Every call to this engine runs in a brand-new subprocess (see
    TTSAdapter._run_engine_subprocess) — there's no in-process cache that
    survives between calls, so without forcing offline mode, EVERY
    synthesis call re-resolves every config/checkpoint file against
    huggingface.co over the network, even though it's already cached
    locally. mlx_audio.tts.utils.load_model() doesn't expose a
    local_files_only-style kwarg, so HF_HUB_OFFLINE forces the same
    behavior transparently — but it MUST be set before importing mlx_audio
    (it's read into a module-level constant at import time, not rechecked
    per call), so we check the cache directory directly first rather than
    importing mlx_audio in a try/except.
    """
    import os

    if _repo_looks_cached(repo):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        logger.info("Not fully cached locally yet — downloading %s…", repo)

    from mlx_audio.tts.utils import load_model

    return load_model(repo)


# ── Qwen3-TTS (zero-shot voice cloning) ─────────────────────────────────────

def _get_qwen3_tts():
    global _qwen3_tts_model
    if _qwen3_tts_model is not None:
        return _qwen3_tts_model
    logger.info("Loading %s…", _QWEN3_TTS_REPO)
    _qwen3_tts_model = _load_model_local_first(_QWEN3_TTS_REPO)
    logger.info("Qwen3-TTS ready.")
    return _qwen3_tts_model


def _synthesise_qwen3_tts(
    text: str, out_path: str, ref_audio: str, ref_text: str, language: str,
) -> tuple[int, float]:
    import numpy as np
    import soundfile as sf

    model = _get_qwen3_tts()
    results = list(model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, language=language or "English"))
    if not results:
        raise RuntimeError("Qwen3-TTS produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


# ── Chatterbox (zero-shot voice cloning, no ref_text needed) ───────────────

def _get_chatterbox():
    global _chatterbox_model
    if _chatterbox_model is not None:
        return _chatterbox_model
    logger.info("Loading %s…", _CHATTERBOX_REPO)
    _chatterbox_model = _load_model_local_first(_CHATTERBOX_REPO)
    logger.info("Chatterbox ready.")
    return _chatterbox_model


def _synthesise_chatterbox(text: str, out_path: str, ref_audio: str) -> tuple[int, float]:
    import numpy as np
    import soundfile as sf

    model = _get_chatterbox()
    kwargs = {"text": text}
    if ref_audio:
        # chatterbox-turbo hard-asserts on this (AssertionError, not a
        # friendly message) — check it ourselves so the failure is clear.
        import soundfile as _sf
        info = _sf.info(ref_audio)
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
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


# ── Voxtral-TTS (preset voices, no cloning) ─────────────────────────────────

_VOXTRAL_VOICES = {
    "casual_male", "casual_female", "cheerful_female", "neutral_male", "neutral_female",
    "fr_male", "fr_female", "es_male", "es_female", "de_male", "de_female",
    "it_male", "it_female", "pt_male", "pt_female", "nl_male", "nl_female",
    "ar_male", "hi_male", "hi_female",
}


def _get_voxtral_tts():
    global _voxtral_tts_model
    if _voxtral_tts_model is not None:
        return _voxtral_tts_model
    logger.info("Loading %s…", _VOXTRAL_TTS_REPO)
    _voxtral_tts_model = _load_model_local_first(_VOXTRAL_TTS_REPO)
    logger.info("Voxtral-TTS ready.")
    return _voxtral_tts_model


def _synthesise_voxtral_tts(text: str, out_path: str, voice: str) -> tuple[int, float]:
    import numpy as np
    import soundfile as sf

    if voice not in _VOXTRAL_VOICES:
        raise ValueError(f"Unknown Voxtral-TTS voice {voice!r}; expected one of {sorted(_VOXTRAL_VOICES)}")
    if voice.startswith("hi_") and text.isascii():
        logger.warning(
            "voice=%s expects Devanagari-script Hindi, but text is pure ASCII "
            "(romanised?) — verified this produces garbled/repetition-looping "
            "audio, not just degraded quality. Pass genuine Devanagari text.",
            voice,
        )

    model = _get_voxtral_tts()
    results = list(model.generate(text=text, voice=voice))
    if not results:
        raise RuntimeError("Voxtral-TTS produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


# ── Higgs-TTS (multilingual, single voice per language) ─────────────────────

def _get_higgs_tts():
    global _higgs_tts_model
    if _higgs_tts_model is not None:
        return _higgs_tts_model
    logger.info("Loading %s…", _HIGGS_TTS_REPO)
    _higgs_tts_model = _load_model_local_first(_HIGGS_TTS_REPO)
    logger.info("Higgs-TTS ready.")
    return _higgs_tts_model


def _synthesise_higgs_tts(
    text: str,
    out_path: str,
    ref_audio: str,
    ref_text: str,
) -> tuple[int, float]:
    import numpy as np
    import soundfile as sf

    model = _get_higgs_tts()
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
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, float(len(audio) / sample_rate)


def main() -> None:
    req = read_request()
    model = req.get("model")
    text = req.get("text", "")
    out_path = req.get("out")

    if not text.strip():
        fail("text is empty")
        return
    if not out_path:
        fail("out path is required")
        return

    try:
        with quiet_stdout():
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            if model == "qwen3-tts":
                sample_rate, duration_s = _synthesise_qwen3_tts(
                    text, out_path, req.get("ref_audio", ""), req.get("ref_text", ""), req.get("language", "English"),
                )
            elif model == "chatterbox":
                sample_rate, duration_s = _synthesise_chatterbox(text, out_path, req.get("ref_audio", ""))
            elif model == "voxtral-tts":
                sample_rate, duration_s = _synthesise_voxtral_tts(text, out_path, req.get("voice", "neutral_female"))
            elif model == "higgs-tts":
                sample_rate, duration_s = _synthesise_higgs_tts(
                    text, out_path,
                    req.get("ref_audio", ""),
                    req.get("ref_text", ""),
                )
            else:
                fail(f"Unknown model: {model!r} (expected qwen3-tts/chatterbox/voxtral-tts/higgs-tts)")
                return
    except Exception as e:
        logger.exception("mlx-audio synthesis failed (model=%s)", model)
        fail(str(e))
        return

    succeed(sample_rate=sample_rate, duration_s=duration_s)


if __name__ == "__main__":
    main()
