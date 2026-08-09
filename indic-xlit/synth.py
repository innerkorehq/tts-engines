#!/usr/bin/env python
"""
IndicXlit transliteration CLI — used by the kokoro engine (indic-to-roman)
and the "hinglish-lid" text pipeline's Route B (roman-to-indic).

Request (stdin JSON):
    {"mode": "indic-to-roman", "text": "..."}
        Convert Indic-script runs (Devanagari, Bengali, Gurmukhi, Gujarati,
        Odia, Tamil, Telugu, Kannada, Malayalam) to Roman-script
        approximations, leaving Latin text/digits/punctuation untouched.
        Response: {"ok": true, "text": "..."}

    {"mode": "roman-to-indic", "words": ["kal", "chalo"], "lang": "hi"}
        Transliterate a batch of Roman-script words (assumed already
        language-tagged as `lang` by an upstream word-LID step — this mode
        does NOT itself decide which words are Hindi) to their native-script
        spelling. Response: {"ok": true, "translations": {"kal": "कल", ...}}
        — a word that fails to transliterate maps to itself unchanged.

Response (stdout JSON): {"ok": false, "error": "..."} on failure.
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


# ── Roman → Indic (used by the "hinglish-lid" text pipeline's Route B) ─────
#
# Opposite direction from above: takes Roman-script words already tagged HI
# by an upstream word-LID step (see tts-engines/text-pipeline/synth.py's
# word-lid engine) and transliterates each into `lang`'s native script.
# Kept as a separate engine cache (`_roman_to_indic_engines`) from
# `_indic_to_roman_engines` above — XlitEngine instances are direction-
# specific (src_script_type="roman" vs "indic"), not interchangeable.

_roman_to_indic_engines: dict[str, object] = {}


def transliterate_roman_to_indic(words: list[str], lang: str) -> dict[str, str]:
    """
    Transliterate each word in `words` from Roman script to `lang`'s native
    script. Returns a dict mapping original -> transliterated word; a word
    that fails to transliterate maps to itself unchanged. Caller (Route B in
    text_pipelines.py) is responsible for only passing words a word-LID step
    actually tagged as belonging to `lang` — this function does no language
    detection of its own.
    """
    engine = _roman_to_indic_engines.get(lang)
    if engine is None:
        engine = _load_xlit_engine(lang, "roman")
        _roman_to_indic_engines[lang] = engine

    out: dict[str, str] = {}
    for word in dict.fromkeys(w for w in words if w and w.strip()):  # dedupe, preserve order
        try:
            result = engine.translit_word(word, lang, topk=1)
        except Exception as e:
            logger.warning("IndicXlit roman→indic failed for %r (lang=%s): %s", word, lang, e)
            out[word] = word
            continue
        if isinstance(result, dict):
            cands = next(iter(result.values()), [])
        elif isinstance(result, list):
            cands = result
        else:
            cands = []
        out[word] = cands[0] if cands else word
    return out


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    req = read_request()
    mode = req.get("mode")

    # ai4bharat-transliteration / pydload print download-progress and status
    # messages directly to stdout via `print()`, which would corrupt our
    # single-line JSON response. Redirect Python-level stdout to stderr for
    # the duration of the work, restoring it only to write the response.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        if mode == "indic-to-roman":
            text = req.get("text", "")
            result_fields = {"text": transliterate_indic_to_roman(text)}
        elif mode == "roman-to-indic":
            words = req.get("words", [])
            lang = req.get("lang", "hi")
            result_fields = {"translations": transliterate_roman_to_indic(words, lang)}
        else:
            sys.stdout = real_stdout
            fail(f"Unknown mode: {mode!r} (expected 'indic-to-roman' or 'roman-to-indic')")
            return
    except Exception as e:
        logger.exception("Transliteration failed")
        sys.stdout = real_stdout
        fail(str(e))
        return
    finally:
        sys.stdout = real_stdout

    succeed(**result_fields)


if __name__ == "__main__":
    main()
