#!/usr/bin/env python
"""
F5-TTS (stock) one-shot synthesis CLI — thin JSON-contract wrapper around
f5tts_infer.py.

Note: TTSAdapter._synthesise_f5tts does NOT call this script. F5-TTS already
runs under its own clean anaconda interpreter (not a uv-managed venv) with a
persistent synthesis daemon (f5tts_server.py) for performance — TTSAdapter
talks to that daemon directly and falls back to f5tts_infer.py one-shot only
if the daemon is unavailable. This synth.py exists for contract-uniformity
and standalone/ad-hoc use, wrapping that same one-shot fallback path.

Request (stdin JSON):
    {"text": "...", "ref_audio": "/path/to/ref.wav", "ref_text": "...", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": 24000, "duration_s": F}
"""
import logging
import subprocess
import sys
from pathlib import Path

from protocol import read_request, succeed, fail

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("f5tts")

_F5TTS_PYTHON = str(Path(__file__).resolve().parent / ".venv" / "bin" / "python")
_F5TTS_FALLBACK_SCRIPT = str(Path(__file__).resolve().parent / "f5tts_infer.py")

_NFE_STEP = 16
_CFG_STRENGTH = 2.0


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    out = req["out"]

    out_path = Path(out)
    out_dir = str(out_path.parent)
    out_file = out_path.name

    proc = subprocess.run(
        [
            _F5TTS_PYTHON, _F5TTS_FALLBACK_SCRIPT,
            "--model", "F5TTS_v1_Base",
            "--ref_audio", ref_audio,
            "--ref_text", ref_text,
            "--gen_text", text,
            "--output_dir", out_dir,
            "--output_file", out_file,
            "--remove_silence", "true",
            "--nfe_step", str(_NFE_STEP),
            "--cfg_strength", str(_CFG_STRENGTH),
        ],
        capture_output=True, text=True,
    )

    # f5-tts exits with code 1 on macOS due to the anaconda PyAV / Homebrew
    # ffmpeg dylib conflict (harmless objc duplicate-class warning). Treat
    # synthesis as successful if the output file was actually written.
    if proc.returncode != 0 and not (out_path.exists() and out_path.stat().st_size > 0):
        raise RuntimeError(f"F5-TTS synthesis failed: {proc.stderr[:3000]}")
    if not out_path.exists():
        raise RuntimeError(f"F5-TTS produced no output at {out}. stderr: {proc.stderr[:3000]}")

    import wave
    with wave.open(out, "rb") as wf:
        sample_rate = wf.getframerate()
        duration_s = wf.getnframes() / sample_rate
    succeed(sample_rate=sample_rate, duration_s=duration_s)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("F5-TTS synthesis failed")
        fail(str(e))
