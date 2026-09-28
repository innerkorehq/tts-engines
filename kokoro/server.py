#!/usr/bin/env python
"""
Kokoro TTS persistent synthesis daemon.

The original `synth.py` CLI is invoked fresh for every narration clip —
meaning the Kokoro `KPipeline` (and the underlying model weights) gets
reloaded from scratch on every single scene. For a render with N scenes
that's N full model loads instead of one.

This script instead loads the KPipeline ONCE and then serves synthesis
requests over a simple line-delimited JSON protocol on stdin/stdout, kept
alive for the lifetime of the worker process — same pattern as
`tts-engines/f5tts/f5tts_server.py`.

Protocol (one JSON object per line, UTF-8, newline-terminated):
    Request  -> {"text": "...", "voice": "af_heart", "speed": 1.0, "out": "/path/to/out.wav"}
    Response <- {"ok": true, "sample_rate": 24000, "duration_s": 1.23}
             <- {"ok": false, "error": "..."}
A "READY" line is written to the protocol stdout once the model has finished
loading, so the parent process knows when it's safe to start sending
requests. A bare "SHUTDOWN" line (not JSON) causes a clean exit.
"""
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path

# ── Protect the protocol stream from library noise ───────────────────────────
# Kokoro/misaki (and libraries it pulls in, e.g. for G2P/model loading) may
# call bare `print(...)` during model load AND during inference, which would
# land on stdout and corrupt our line-delimited JSON protocol if left alone.
# So: duplicate the *original* stdout fd into a dedicated file object
# reserved exclusively for protocol messages (READY / JSON responses), then
# repoint `sys.stdout` at stderr so every incidental `print()` from imported
# libraries is harmless diagnostic noise instead of protocol corruption.
# This must happen BEFORE importing any ML libraries (kokoro, numpy, etc.).
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("kokoro_server")

_TARGET_SR = 24_000


def _log(msg: str) -> None:
    """Diagnostic logging to stderr — stdout is reserved for the protocol."""
    print(f"[kokoro_server] {msg}", file=sys.stderr, flush=True)


# ── eSpeak phoneme code support via Kokoro/misaki ───────────────────────────
#
# Pronunciation entries may use eSpeak-ng phoneme codes in [[...]] syntax,
# e.g. [[p'Eks@lz]] for "Pexels".  Kokoro's misaki G2P does not interpret
# these natively, so we:
#   1. Split the utterance on [[...]] boundaries.
#   2. Synthesise plain-text spans via the normal Kokoro pipeline.
#   3. Convert [[...]] spans to IPA using the eSpeak→IPA table, then pass
#      the IPA string directly to KPipeline.generate_from_tokens() — which
#      synthesises using the same Kokoro voice model, no extra process.
#   4. Concatenate all 24 kHz PCM segments.

_PHONEME_PATTERN = re.compile(r"(\[\[.*?\]\])", re.DOTALL)

# eSpeak notation → IPA mapping. Order matters for the alternation regex —
# multi-char sequences (diphthongs, affricates) must be tried before
# single-char ones so they're never split apart.
_ESPEAK_MAP: dict = {
    # Diphthongs
    "eI": "eɪ", "aI": "aɪ", "OI": "ɔɪ", "@U": "oʊ", "aU": "aʊ",
    "I@": "ɪə", "e@": "ɛə", "U@": "ʊə",
    # Affricates
    "tS": "tʃ", "dZ": "dʒ",
    # Multi-char consonants
    "T": "θ", "D": "ð", "S": "ʃ", "Z": "ʒ", "N": "ŋ",
    # Vowels (uppercase before lowercase to avoid partial match issues)
    "I": "ɪ", "E": "ɛ", "V": "ʌ", "A": "ɑ", "O": "ɒ", "U": "ʊ",
    "@": "ə", "3": "ɜ", "&": "æ",
    # Base consonants
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "f": "f", "v": "v", "h": "h", "m": "m", "n": "n",
    "l": "l", "r": "ɹ", "w": "w", "j": "j",
    # Base vowels
    "i": "i", "u": "u", "e": "e", "o": "o", "a": "a",
    # Stress markers
    "'": "ˈ", ",": "ˌ",
    # Other
    "?": "ʔ", "x": "x",
}
_ESPEAK_RE = re.compile("|".join(re.escape(k) for k in _ESPEAK_MAP))


def _has_phoneme_codes(text: str) -> bool:
    return "[[" in text and "]]" in text


def _split_phoneme_segments(text: str):
    """
    Split *text* into alternating (kind, value) pairs.

    Example:
        "Check out [[p'Eks@lz]] today"
        → [("text", "Check out "), ("phoneme", "[[p'Eks@lz]]"), ("text", " today")]
    """
    segments = []
    for part in _PHONEME_PATTERN.split(text):
        if not part:
            continue
        kind = "phoneme" if (part.startswith("[[") and part.endswith("]]")) else "text"
        segments.append((kind, part))
    return segments


def _espeak_to_ipa(code: str) -> str:
    """Convert the inner content of [[...]] (eSpeak notation) to IPA."""
    return _ESPEAK_RE.sub(lambda m: _ESPEAK_MAP.get(m.group(), m.group()), code)


_REPO = "hexgrad/Kokoro-82M"


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or anything that imports it,
    e.g. kokoro below) before HF_HUB_OFFLINE is set makes the env var a
    no-op (huggingface_hub reads it into a module-level constant at import
    time, not dynamically per call). Same pattern as
    chatterbox-multilingual-hi/server.py's identical helper.
    """
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _load_pipeline():
    # One-time setup: import + construct the Kokoro KPipeline. This is the
    # expensive step (may trigger a model download on first run) that the
    # original synth.py paid on every single invocation via its lazy
    # module-global `_get_pipeline()`. Here it happens exactly once, before
    # READY is signalled, and the resulting object is reused for every
    # subsequent request for the lifetime of this process.
    #
    # kokoro's KPipeline uses huggingface_hub.hf_hub_download() unconditionally
    # for every file it needs (config/weights/voice packs) — no local-cache
    # check of its own, so without HF_HUB_OFFLINE it hits huggingface.co's API
    # on every single daemon startup even when everything is already cached,
    # and a transient HF Hub connectivity blip there crashes startup outright
    # (confirmed in production: local-chat-mlx hit exactly this). Force
    # offline mode once the repo is confirmed cached locally.
    if _repo_looks_cached(_REPO):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        _log(f"Not fully cached locally yet — downloading {_REPO}…")

    from kokoro import KPipeline
    _log("Loading Kokoro KPipeline (one-time, first call may trigger model download)…")
    pipeline = KPipeline(lang_code="a")
    _log("Kokoro KPipeline ready.")
    return pipeline


def _synthesise_plain(pipeline, text: str, voice: str, speed: float, out: str):
    import numpy as np
    import soundfile as sf

    audio_chunks = []
    for result in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(result.audio)

    if not audio_chunks:
        raise RuntimeError("Kokoro returned no audio chunks for the provided text.")

    audio = np.concatenate(audio_chunks)
    sf.write(out, audio, _TARGET_SR)
    return _TARGET_SR, len(audio) / _TARGET_SR


def _synthesise_mixed(pipeline, text: str, voice: str, speed: float, out: str):
    """
    Segment synthesis for text that contains [[eSpeak phoneme codes]].

    Plain-text spans go through the standard Kokoro pipeline. Phoneme spans
    are converted to IPA and passed directly to
    KPipeline.generate_from_tokens() — same Kokoro voice model, no
    subprocess, no sample-rate mismatch.
    """
    import numpy as np
    import soundfile as sf

    pcm_parts = []

    for kind, value in _split_phoneme_segments(text):
        chunks = []

        if kind == "phoneme":
            inner = value[2:-2].strip()
            ipa = _espeak_to_ipa(inner)
            if not ipa:
                continue
            logger.debug("eSpeak [[%s]] -> IPA '%s'", inner, ipa)
            for result in pipeline.generate_from_tokens(ipa, voice=voice, speed=speed):
                if result.audio is not None:
                    chunks.append(result.audio)
        else:
            stripped = value.strip()
            if not stripped:
                continue
            for result in pipeline(stripped, voice=voice, speed=speed):
                if result.audio is not None:
                    chunks.append(result.audio)

        if chunks:
            pcm_parts.append(np.concatenate(chunks))

    if not pcm_parts:
        raise RuntimeError(f"No audio produced for text: {text!r}")

    audio = np.concatenate(pcm_parts)
    sf.write(out, audio, _TARGET_SR)
    return _TARGET_SR, len(audio) / _TARGET_SR


def _handle_request(req: dict, pipeline) -> dict:
    text = req.get("text", "")
    voice = req.get("voice", "af_heart")
    speed = req.get("speed", 1.0)
    out = req["out"]

    if _has_phoneme_codes(text):
        sample_rate, duration_s = _synthesise_mixed(pipeline, text, voice, speed, out)
    else:
        sample_rate, duration_s = _synthesise_plain(pipeline, text, voice, speed, out)

    return {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}


def main() -> int:
    pipeline = _load_pipeline()

    # Signal readiness to the parent process on the protected protocol stream
    # (NOT via `print`, which now points at stderr).
    _send("READY")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "SHUTDOWN":
            _log("Received SHUTDOWN — exiting.")
            return 0
        try:
            req = json.loads(line)
            resp = _handle_request(req, pipeline)
        except Exception as exc:  # noqa: BLE001 — must always answer the request
            _log(f"Error handling request: {exc}\n{traceback.format_exc()}")
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(json.dumps(resp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
