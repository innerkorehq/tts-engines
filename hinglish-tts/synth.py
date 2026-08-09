#!/usr/bin/env python
"""
hinglish-tts CLI — thin protocol wrapper around this vendored repo's own
inference.py (harrrshall/hinglish-tts), used exactly per its documented API:

    from inference import load_model, synthesize

No reimplementation — inference.py and scoring/scripts/lib_normalize.py are
vendored verbatim from the upstream repo; this file only adapts them to
vidgen's stdin/stdout JSON subprocess contract (see protocol.py).

Request (stdin JSON):
    {"text": "...", "ref_audio": "/path/to/ref.wav", "ref_text": "...", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": 24000, "duration_s": F}
"""
import logging
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("hinglish-tts")


def _apply_fairseq_compat_shims() -> None:
    """
    fairseq==0.12.2 (pulled in by ai4bharat-transliteration, used by the
    vendored scoring/scripts/lib_normalize.py) needs two scoped compat
    shims on Python 3.11 / torch>=2.6 — same as tts-engines/indic-xlit/
    synth.py's _load_xlit_engine. Applied here, once, before any inference
    — the vendored upstream files themselves are left untouched.
    """
    import dataclasses as _dataclasses
    import torch as _torch

    orig_get_field = _dataclasses._get_field

    def _get_field_compat(cls_, a_name, a_type, default_kw_only, *args, **kwargs):
        default = getattr(cls_, a_name, _dataclasses.MISSING)
        target = default.default if isinstance(default, _dataclasses.Field) else default
        if target is not _dataclasses.MISSING and target.__class__.__hash__ is None:
            try:
                target.__class__.__hash__ = object.__hash__
            except TypeError:
                pass
        return orig_get_field(cls_, a_name, a_type, default_kw_only, *args, **kwargs)

    orig_torch_load = _torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    _dataclasses._get_field = _get_field_compat
    _torch.load = _torch_load_compat


_apply_fairseq_compat_shims()

from inference import load_model, synthesize  # noqa: E402 — must follow compat shims

_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading IndicF5 via inference.load_model()…")
        _model = load_model()
        logger.info("IndicF5 ready.")
    return _model


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    out = req["out"]

    import soundfile as sf

    with quiet_stdout():
        model = _get_model()
        audio = synthesize(model, text, ref_audio_path=ref_audio, ref_text=ref_text)
        sample_rate = 24000
        sf.write(out, audio, samplerate=sample_rate)

    succeed(sample_rate=sample_rate, duration_s=len(audio) / sample_rate)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("hinglish-tts synthesis failed")
        fail(str(e))
