#!/usr/bin/env python
"""
mlx-audio persistent synthesis daemon.

`synth.py` in this directory handles 6 sub-model variants (qwen3-tts,
chatterbox, chatterbox-multilingual, voxtral-tts, higgs-tts, svara-tts) but is
invoked as a *fresh* subprocess per request — every synthesis call re-downloads
nothing (thanks to HF_HUB_OFFLINE) but still re-loads that model's weights
from disk onto the GPU/Neural Engine from scratch. For a render with many
scenes using the same voice, that's one full model load per scene instead of
one for the whole render.

This script instead loads models LAZILY (on first request, since we don't
know which of the 6 sub-models will be requested first) and then serves
requests over the same line-delimited JSON protocol on stdin/stdout used by
the other *_server.py daemons in this repo, kept alive for the worker
process's lifetime.

Single-model-resident design (deliberate, not an oversight): mlx-audio models
here range from ~1-4GB+ each; keeping all 6 loaded simultaneously would multiply
memory footprint for no benefit in the common case (a render typically uses
one voice/model repeatedly). So this daemon keeps AT MOST ONE sub-model's
weights resident at a time:
  - First request for model X: load X, remember it as current.
  - Next request, same model X: reuse the already-loaded object (warm path).
  - Next request, different model Y: drop the reference to X's model object,
    run gc.collect() (mlx-audio/MLX exposes no explicit "unload" hook we could
    find — this relies on Python GC + MLX's own lazy buffer reclamation), then
    load Y fresh and make it current.
This trades "switching voices mid-render costs a reload" for "6x lower peak
memory" — acceptable since voice switches are rare relative to same-voice
reuse within one render.

Protocol (one JSON object per line, UTF-8, newline-terminated):
    Request  -> {"model": "qwen3-tts", "text": "...", "out": "/abs/out.wav", ...}
    Response <- {"ok": true, "sample_rate": N, "duration_s": F}
             <- {"ok": false, "error": "..."}
A "READY" line is written to the protocol stream as soon as the process has
finished importing — NOT after loading any model, since which of the 6
sub-models will be requested first isn't known at startup. The first real
model load happens lazily on the first request that needs it, same as
synth.py's per-model `_get_*()` memoization, just kept across requests instead
of re-run every process invocation.
A "SHUTDOWN" line (bare string, not JSON) exits cleanly.
"""
import gc
import json
import logging
import os
import sys
import traceback
from pathlib import Path

# ── Protect the protocol stream from library noise ───────────────────────────
# mlx-audio (and libraries it pulls in, e.g. huggingface_hub download
# progress, mlx itself) commonly print progress/status directly via bare
# `print(...)` during model loading AND during inference — this would corrupt
# our line-delimited JSON protocol if left on stdout. So: duplicate the
# *original* stdout fd into a dedicated file object reserved exclusively for
# protocol messages (READY / JSON responses), then repoint `sys.stdout` at
# stderr so incidental `print()` calls become harmless diagnostic noise
# instead of protocol corruption. This must happen BEFORE importing anything
# ML-related.
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mlx-audio-server")


def _log(msg: str) -> None:
    """Diagnostic logging to stderr — stdout is reserved for the protocol."""
    print(f"[mlx-audio-server] {msg}", file=sys.stderr, flush=True)


# ── Model repo IDs (mirrors synth.py) ────────────────────────────────────────
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
_current_model_name: str | None = None
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
    documented way to let their backing memory be released. If a future
    mlx-audio/mlx version adds an explicit release hook, prefer that here.
    """
    global _current_model_obj, _current_model_name
    if _current_model_obj is not None:
        _log(f"Unloading current model ({_current_model_name}) before switching…")
    _current_model_obj = None
    _current_model_name = None
    gc.collect()
    try:
        import mlx.core as mx
        # Best-effort: release cached (but unreferenced) buffers in MLX's
        # memory pool back to the system allocator. Safe no-op if the
        # installed mlx version lacks this call.
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

    _log(f"Loading {model_name} ({repo})…")
    model_obj = _load_model_local_first(repo)
    _log(f"{model_name} ready.")

    _current_model_obj = model_obj
    _current_model_name = model_name
    return model_obj


# ── Per-model inference (mirrors synth.py's _synthesise_* functions) ────────

def _synthesise_qwen3_tts(model, text: str, out_path: str, ref_audio: str, ref_text: str, language: str):
    import numpy as np
    import soundfile as sf

    results = list(model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, language=language or "English"))
    if not results:
        raise RuntimeError("Qwen3-TTS produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


def _synthesise_chatterbox(model, text: str, out_path: str, ref_audio: str):
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
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


def _synthesise_chatterbox_multilingual(model, text: str, out_path: str, ref_audio: str, language: str):
    import numpy as np
    import soundfile as sf

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
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


def _synthesise_voxtral_tts(model, text: str, out_path: str, voice: str):
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

    results = list(model.generate(text=text, voice=voice))
    if not results:
        raise RuntimeError("Voxtral-TTS produced no audio")
    audio = np.concatenate([np.array(r.audio) for r in results])
    sample_rate = getattr(results[0], "sample_rate", _TARGET_SR)
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


def _synthesise_svara_tts(model, text: str, out_path: str, voice: str):
    import numpy as np
    import soundfile as sf

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
    sf.write(out_path, audio, sample_rate)
    return sample_rate, len(audio) / sample_rate


def _synthesise_higgs_tts(model, text: str, out_path: str, ref_audio: str, ref_text: str):
    import numpy as np
    import soundfile as sf

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


def _handle_request(req: dict) -> dict:
    model_name = req.get("model")
    text = req.get("text", "")
    out_path = req.get("out")

    if model_name not in _KNOWN_MODELS:
        raise ValueError(
            f"Unknown model: {model_name!r} (expected one of {sorted(_KNOWN_MODELS)})"
        )
    if not text.strip():
        raise ValueError("text is empty")
    if not out_path:
        raise ValueError("out path is required")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    model = _ensure_model_loaded(model_name)

    if model_name == "qwen3-tts":
        sample_rate, duration_s = _synthesise_qwen3_tts(
            model, text, out_path, req.get("ref_audio", ""), req.get("ref_text", ""), req.get("language", "English"),
        )
    elif model_name == "chatterbox":
        sample_rate, duration_s = _synthesise_chatterbox(model, text, out_path, req.get("ref_audio", ""))
    elif model_name == "chatterbox-multilingual":
        sample_rate, duration_s = _synthesise_chatterbox_multilingual(
            model, text, out_path, req.get("ref_audio", ""), req.get("language", "en"),
        )
    elif model_name == "voxtral-tts":
        sample_rate, duration_s = _synthesise_voxtral_tts(model, text, out_path, req.get("voice", "neutral_female"))
    elif model_name == "higgs-tts":
        sample_rate, duration_s = _synthesise_higgs_tts(
            model, text, out_path, req.get("ref_audio", ""), req.get("ref_text", ""),
        )
    elif model_name == "svara-tts":
        sample_rate, duration_s = _synthesise_svara_tts(model, text, out_path, req.get("voice", "Hindi (Female)"))
    else:  # unreachable — guarded by _KNOWN_MODELS check above
        raise AssertionError(f"unhandled model {model_name!r}")

    return {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}


def main() -> int:
    # No model is loaded yet — which of the 6 sub-models is needed isn't
    # known until the first request arrives, so signal readiness as soon as
    # imports/setup are done rather than blocking on a model load.
    _send("READY")
    _log("Ready — no model loaded yet, will load lazily on first request.")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "SHUTDOWN":
            _log("Received SHUTDOWN — exiting.")
            return 0
        try:
            req = json.loads(line)
            resp = _handle_request(req)
        except Exception as exc:  # noqa: BLE001 — must always answer the request
            _log(f"Error handling request: {exc}\n{traceback.format_exc()}")
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(json.dumps(resp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
