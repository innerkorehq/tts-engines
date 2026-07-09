#!/usr/bin/env python
"""
IndicXlit transliteration CLI — shared by the kokoro and indicf5-hinglish
engines.

Request (stdin JSON):
    {"mode": "indic-to-roman", "text": "..."}
        Convert Indic-script runs (Devanagari, Bengali, Gurmukhi, Gujarati,
        Odia, Tamil, Telugu, Kannada, Malayalam) to Roman-script
        approximations, leaving Latin text/digits/punctuation untouched.

    {"mode": "hinglish-to-devanagari", "text": "..."}
        Convert mixed-script Hinglish (Devanagari / Roman-Hindi / English) to
        unified Devanagari.

Response (stdout JSON): {"ok": true, "text": "..."}  or  {"ok": false, "error": "..."}
"""
import logging
import re
import sys

from protocol import read_request, succeed, fail

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("indic-xlit")


# ── IndicXlit engine loading (with fairseq/torch compat shims) ──────────────

def _load_xlit_engine(lang: str, src_script_type: str):
    """
    Load an ai4bharat-transliteration XlitEngine, with two scoped compat
    shims for fairseq==0.12.2 on Python 3.11 / torch>=2.6:

    1. fairseq's dataclass configs use other dataclass *instances* as field
       defaults (e.g. `common: CommonConfig = CommonConfig()`). Python 3.11's
       dataclasses module rejects any default whose class has __hash__ = None
       ("mutable default ... is not allowed"). Give such classes a real
       (identity-based) __hash__ just for this import.
    2. fairseq's checkpoint_utils.load_checkpoint_to_cpu() calls
       `torch.load(f, map_location=...)` with no `weights_only` argument.
       torch>=2.6 defaults to `weights_only=True`, which refuses to unpickle
       the `argparse.Namespace` objects in IndicXlit's checkpoint (ai4bharat's
       own pretrained model — trusted). Default to `weights_only=False` only
       when the caller didn't specify it.
    """
    import dataclasses as _dataclasses
    import torch as _torch

    _orig_get_field = _dataclasses._get_field

    def _get_field_compat(cls_, a_name, a_type, default_kw_only, *args, **kwargs):
        default = getattr(cls_, a_name, _dataclasses.MISSING)
        target = default.default if isinstance(default, _dataclasses.Field) else default
        if target is not _dataclasses.MISSING and target.__class__.__hash__ is None:
            try:
                target.__class__.__hash__ = object.__hash__
            except TypeError:
                pass
        return _orig_get_field(cls_, a_name, a_type, default_kw_only, *args, **kwargs)

    _orig_torch_load = _torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    _dataclasses._get_field = _get_field_compat
    _torch.load = _torch_load_compat
    try:
        from ai4bharat.transliteration import XlitEngine
        logger.info("Loading IndicXlit (lang=%s, src_script_type=%s)…", lang, src_script_type)
        engine = XlitEngine(lang, beam_width=4, src_script_type=src_script_type)
        logger.info("IndicXlit ready.")
        return engine
    finally:
        _dataclasses._get_field = _orig_get_field
        _torch.load = _orig_torch_load


# ── Indic → Roman (used by kokoro for non-English-script narration) ────────

# Unicode block → IndicXlit language code, ordered for first-match lookup.
_INDIC_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "hi"),  # Devanagari (Hindi, Marathi, Sanskrit, ...)
    (0x0980, 0x09FF, "bn"),  # Bengali / Assamese
    (0x0A00, 0x0A7F, "pa"),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF, "gu"),  # Gujarati
    (0x0B00, 0x0B7F, "or"),  # Odia
    (0x0B80, 0x0BFF, "ta"),  # Tamil
    (0x0C00, 0x0C7F, "te"),  # Telugu
    (0x0C80, 0x0CFF, "kn"),  # Kannada
    (0x0D00, 0x0D7F, "ml"),  # Malayalam
)

_indic_to_roman_engines: dict[str, object] = {}
_INDIC_TRANSLIT_TOKEN_RE = re.compile(r"([A-Za-z]+|[ऀ-ൿ]+|\d+|\s+|.)", re.DOTALL)


def _script_lang_for_char(ch: str) -> "str | None":
    cp = ord(ch)
    for lo, hi, lang in _INDIC_SCRIPT_RANGES:
        if lo <= cp <= hi:
            return lang
    return None


def transliterate_indic_to_roman(text: str) -> str:
    if not text or not any(_script_lang_for_char(c) for c in text):
        return text

    out = []
    for m in _INDIC_TRANSLIT_TOKEN_RE.finditer(text):
        tok = m.group(0)
        lang = _script_lang_for_char(tok[0])
        if lang is None:
            out.append(tok)
            continue
        engine = _indic_to_roman_engines.get(lang)
        if engine is None:
            engine = _load_xlit_engine(lang, "indic")
            _indic_to_roman_engines[lang] = engine
        try:
            result = engine.translit_word(tok, lang, topk=1)
        except Exception as e:
            logger.warning("IndicXlit indic→roman failed for %r (lang=%s): %s", tok, lang, e)
            out.append(tok)
            continue
        if isinstance(result, dict):
            cands = next(iter(result.values()), [])
        elif isinstance(result, list):
            cands = result
        else:
            cands = []
        out.append(cands[0] if cands else tok)
    return "".join(out)


# ── Hinglish → Devanagari (used by indicf5-hinglish) ────────────────────────

# Short Roman-script Hindi function words IndicXlit mis-transliterates as
# English phonetics (e.g. "mai" → "माई" instead of "मैं"). Checked first.
_HINGLISH_FUNCTION_WORDS: dict[str, str] = {
    "aa":  "आ",    # "come" / leading vowel — IndicXlit gives "एए"
    "hu":  "हूं",   # "am" (1st person) — IndicXlit gives "हू" (drops nasal)
    "mai": "मैं",   # "I" — IndicXlit gives "माई" (English "my")
    "tu":  "तू",    # "you" (informal) — IndicXlit gives "टू" (English "to")
}

# English loanwords common in Hinglish — stable canonical Devanagari forms.
_HINGLISH_LOANWORDS: dict[str, str] = {
    "biryani": "बिरयानी", "boss": "बॉस", "butter": "बटर", "cancel": "कैंसल",
    "chicken": "चिकन", "close": "क्लोज़", "deadline": "डेडलाइन",
    "deliver": "डिलीवर", "event": "इवेंट", "file": "फाइल",
    "homework": "होमवर्क", "issue": "इश्यू", "laptop": "लैपटॉप",
    "leave": "लीव", "log": "लॉग", "lucky": "लकी", "lunch": "लंच",
    "message": "मैसेज", "movie": "मूवी", "nervous": "नर्वस", "new": "न्यू",
    "office": "ऑफिस", "party": "पार्टी", "personal": "पर्सनल",
    "place": "प्लेस", "please": "प्लीज़", "presentation": "प्रेज़ेंटेशन",
    "quickly": "क्विकली", "reply": "रिप्लाई", "restaurant": "रेस्टोरेंट",
    "review": "रिव्यू", "ticket": "टिकट", "tomorrow": "टुमॉरो",
    "try": "ट्राई", "wait": "वेट", "waste": "वेस्ट",
}

# Indian named entities (cities, people, brands) — canonical Devanagari.
# Apostrophes are split off before lookup, e.g. "Karim's" → "karim".
_HINGLISH_NAMED_ENTITIES: dict[str, str] = {
    "aishwarya": "ऐश्वर्या", "arjun": "अर्जुन", "bengaluru": "बेंगलुरु",
    "chennai": "चेन्नई", "connaught": "कनॉट", "consultancy": "कंसल्टेंसी",
    "delhi": "दिल्ली", "hyderabad": "हैदराबाद", "karim": "करीम",
    "khanna": "खन्ना", "mr": "मिस्टर", "mumbai": "मुंबई", "old": "ओल्ड",
    "paradise": "पैराडाइज़", "priya": "प्रिया", "pune": "पुणे",
    "rohan": "रोहन", "services": "सर्विसेज़", "tata": "टाटा",
}

_HINGLISH_CANONICAL: dict[str, str] = {
    **_HINGLISH_FUNCTION_WORDS,
    **_HINGLISH_LOANWORDS,
    **_HINGLISH_NAMED_ENTITIES,
}

# Captures: ASCII alpha runs | Devanagari runs | digit runs | whitespace runs
# | any single char (punctuation/apostrophes). Apostrophes are NOT folded
# into ASCII-alpha tokens, so "Karim's" → ["Karim", "'", "s"] and "karim"
# hits the named-entity table.
_HINGLISH_TOKEN_RE = re.compile(r"([A-Za-z]+|[ऀ-ॿ]+|\d+|\s+|.)", re.DOTALL)

_hinglish_xlit_engine = None


def _get_hinglish_xlit():
    global _hinglish_xlit_engine
    if _hinglish_xlit_engine is None:
        _hinglish_xlit_engine = _load_xlit_engine("hi", "en")
    return _hinglish_xlit_engine


def _is_devanagari_token(tok: str) -> bool:
    has_deva = any("ऀ" <= c <= "ॿ" for c in tok)
    has_ascii_alpha = any(c.isascii() and c.isalpha() for c in tok)
    return has_deva and not has_ascii_alpha


def _is_ascii_alpha_token(tok: str) -> bool:
    return bool(tok) and all(c.isascii() and (c.isalpha() or c == "'") for c in tok)


def _normalize_ascii_token(tok: str) -> str:
    key = tok.lower()
    canonical = _HINGLISH_CANONICAL.get(key)
    if canonical is not None:
        return canonical
    xlit = _get_hinglish_xlit()
    result = xlit.translit_word(key, topk=1)
    if isinstance(result, dict):
        cands = result.get("hi") or next(iter(result.values()), [])
    elif isinstance(result, list):
        cands = result
    else:
        cands = []
    return cands[0] if cands else tok


def hinglish_to_devanagari(text: str) -> str:
    """Convert Hinglish text to unified Devanagari for IndicF5 input. Idempotent."""
    if not text:
        return text

    out = []
    for m in _HINGLISH_TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            out.append(tok)
        elif _is_devanagari_token(tok):
            out.append(tok)
        elif _is_ascii_alpha_token(tok):
            out.append(_normalize_ascii_token(tok))
        else:
            out.append(tok)
    return "".join(out)


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    req = read_request()
    mode = req.get("mode")
    text = req.get("text", "")

    # ai4bharat-transliteration / pydload print download-progress and status
    # messages directly to stdout via `print()`, which would corrupt our
    # single-line JSON response. Redirect Python-level stdout to stderr for
    # the duration of the work, restoring it only to write the response.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        if mode == "indic-to-roman":
            result = transliterate_indic_to_roman(text)
        elif mode == "hinglish-to-devanagari":
            result = hinglish_to_devanagari(text)
        else:
            sys.stdout = real_stdout
            fail(f"Unknown mode: {mode!r} (expected 'indic-to-roman' or 'hinglish-to-devanagari')")
            return
    except Exception as e:
        logger.exception("Transliteration failed")
        sys.stdout = real_stdout
        fail(str(e))
        return
    finally:
        sys.stdout = real_stdout

    succeed(text=result)


if __name__ == "__main__":
    main()
