#!/usr/bin/env python
"""
IndicF5-Hinglish synthesis CLI.

Mixed-script "Hinglish" narration (Devanagari + romanised Hindi + English,
often in the same sentence) confuses IndicF5: it expects one script. We
normalise the whole utterance to unified Devanagari (by shelling out to the
tts-engines/indic-xlit subproject) before handing it to the IndicF5 model
(shares model-loading code with the `indicf5` engine).

Ported from https://github.com/harrrshall/hinglish-tts
(scoring/scripts/lib_normalize.py — `to_unified_devanagari`).

Request (stdin JSON):
    {"text": "...", "ref_audio": "/path/to/ref.wav", "ref_text": "...", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": 24000, "duration_s": F}
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

# IndicF5 model-loading/synthesis code lives in the sibling indicf5 engine —
# import it directly from its source file (each engine venv is isolated, but
# importing a single sibling .py file by path has no dependency-resolution
# implications, only the symbols actually used need to import cleanly under
# THIS venv's deps, which match indicf5's).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "indicf5"))

from protocol import read_request, succeed, fail, quiet_stdout  # noqa: E402
from synth import load_model, synthesise  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("indicf5-hinglish")

_INDIC_XLIT_DIR = Path(__file__).resolve().parents[1] / "indic-xlit"


def hinglish_to_devanagari(text: str) -> str:
    venv_py = _INDIC_XLIT_DIR / ".venv" / "bin" / "python"
    proc = subprocess.run(
        [str(venv_py), str(_INDIC_XLIT_DIR / "synth.py")],
        input=json.dumps({"mode": "hinglish-to-devanagari", "text": text}),
        capture_output=True, text=True,
    )
    response = json.loads(proc.stdout or "{}")
    if proc.returncode != 0 or not response.get("ok"):
        raise RuntimeError(f"indic-xlit transliteration failed: {response.get('error') or proc.stderr[:2000]}")
    return response["text"]


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    out = req["out"]

    normalized = hinglish_to_devanagari(text)
    if not normalized.strip():
        raise ValueError(f"Text became empty after Hinglish normalisation: {text!r}")
    logger.debug("Hinglish normalisation: %r -> %r", text, normalized)

    with quiet_stdout():
        model = load_model()
        sample_rate, duration_s = synthesise(model, normalized, ref_audio, ref_text, out)
    succeed(sample_rate=sample_rate, duration_s=duration_s)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("IndicF5-Hinglish synthesis failed")
        fail(str(e))
