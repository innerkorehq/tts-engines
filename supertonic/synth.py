#!/usr/bin/env python
"""
Supertonic TTS 3 synthesis CLI — ONNX-based, 44.1kHz, 31-language, CPU-only.

Request (stdin JSON):
    {
      "text":  "...",
      "out":   "/abs/path/out.wav",
      "voice": "M1",          # M1-M5 | F1-F5 (default "M1")
      "lang":  "en",          # 31-language code or "na" (language-agnostic, default "na")
      "speed": 1.05,          # 0.7–2.0 (default 1.05)
      "steps": 8              # quality 5–12 (default 8)
    }

Response (stdout JSON): {"ok": true, "sample_rate": 44100, "duration_s": F}
                        {"ok": false, "error": "..."} on failure.
"""
import logging
import sys

import soundfile as sf

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("supertonic")

_SAMPLE_RATE = 44_100

_tts = None


def _get_tts():
    global _tts
    if _tts is None:
        from supertonic import TTS
        logger.info("Loading Supertonic TTS 3 (first call — model download may occur)…")
        with quiet_stdout():
            _tts = TTS(auto_download=True)
        logger.info("Supertonic TTS 3 ready.")
    return _tts


def main() -> None:
    req = read_request()
    text  = req.get("text", "")
    out   = req.get("out", "")
    voice = req.get("voice") or "M1"
    lang  = req.get("lang")  or "na"
    speed = float(req.get("speed") or 1.05)
    steps = int(req.get("steps") or 8)

    if not text.strip():
        fail("text is required")
        return
    if not out:
        fail("out path is required")
        return

    try:
        tts = _get_tts()
        with quiet_stdout():
            style = tts.get_voice_style(voice_name=voice)
            wav, duration = tts.synthesize(
                text=text,
                lang=lang,
                voice_style=style,
                total_steps=steps,
                speed=speed,
            )
        # wav: float32 numpy array (1, num_samples); trim to actual duration
        samples = wav[0, : int(_SAMPLE_RATE * duration[0].item())]
        sf.write(out, samples, _SAMPLE_RATE)
        duration_s = float(len(samples) / _SAMPLE_RATE)
        logger.info("Synthesized %.2fs → %s", duration_s, out)
    except Exception as e:
        logger.exception("Supertonic synthesis failed")
        fail(str(e))
        return

    succeed(sample_rate=_SAMPLE_RATE, duration_s=duration_s)


if __name__ == "__main__":
    main()
