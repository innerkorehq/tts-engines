#!/usr/bin/env python
"""
Kokoro TTS synthesis CLI (incl. eSpeak [[phoneme]] code support).

Request (stdin JSON):
    {"text": "...", "voice": "af_heart", "speed": 1.0, "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": 24000, "duration_s": F}
"""
import logging
import re
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("kokoro")

_TARGET_SR = 24_000

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        logger.info("Loading Kokoro KPipeline (first call — model download may occur)...")
        _pipeline = KPipeline(lang_code="a")
        logger.info("Kokoro KPipeline ready.")
    return _pipeline


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
_ESPEAK_MAP: dict[str, str] = {
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


def _split_phoneme_segments(text: str) -> list[tuple[str, str]]:
    """
    Split *text* into alternating (kind, value) pairs.

    Example:
        "Check out [[p'Eks@lz]] today"
        → [("text", "Check out "), ("phoneme", "[[p'Eks@lz]]"), ("text", " today")]
    """
    segments: list[tuple[str, str]] = []
    for part in _PHONEME_PATTERN.split(text):
        if not part:
            continue
        kind = "phoneme" if (part.startswith("[[") and part.endswith("]]")) else "text"
        segments.append((kind, part))
    return segments


def _espeak_to_ipa(code: str) -> str:
    """Convert the inner content of [[...]] (eSpeak notation) to IPA."""
    return _ESPEAK_RE.sub(lambda m: _ESPEAK_MAP.get(m.group(), m.group()), code)


def synthesise_plain(text: str, voice: str, speed: float, out: str) -> tuple[int, float]:
    import numpy as np
    import soundfile as sf

    pipeline = _get_pipeline()
    audio_chunks = []
    for result in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(result.audio)

    if not audio_chunks:
        raise RuntimeError("Kokoro returned no audio chunks for the provided text.")

    audio = np.concatenate(audio_chunks)
    sf.write(out, audio, _TARGET_SR)
    return _TARGET_SR, len(audio) / _TARGET_SR


def synthesise_mixed(text: str, voice: str, speed: float, out: str) -> tuple[int, float]:
    """
    Segment synthesis for text that contains [[eSpeak phoneme codes]].

    Plain-text spans go through the standard Kokoro pipeline. Phoneme spans
    are converted to IPA and passed directly to
    KPipeline.generate_from_tokens() — same Kokoro voice model, no
    subprocess, no sample-rate mismatch.
    """
    import numpy as np
    import soundfile as sf

    pipeline = _get_pipeline()
    pcm_parts: list[np.ndarray] = []

    for kind, value in _split_phoneme_segments(text):
        chunks: list[np.ndarray] = []

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


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    voice = req.get("voice", "af_heart")
    speed = req.get("speed", 1.0)
    out = req["out"]

    with quiet_stdout():
        if _has_phoneme_codes(text):
            sample_rate, duration_s = synthesise_mixed(text, voice, speed, out)
        else:
            sample_rate, duration_s = synthesise_plain(text, voice, speed, out)

    succeed(sample_rate=sample_rate, duration_s=duration_s)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Kokoro synthesis failed")
        fail(str(e))
