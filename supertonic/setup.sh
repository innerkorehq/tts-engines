#!/usr/bin/env bash
# Sets up the isolated venv for the Supertonic TTS 3 engine. Safe to re-run.
#
# Supertonic is pure-Python ONNX — no build-from-source steps, no GPU required.
# Model weights download from HuggingFace automatically on first synthesis call
# (or `supertonic download` can pre-fetch them).
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "supertonic venv ready: tts-engines/supertonic/.venv"
echo "Model weights (~500MB) download from HuggingFace on first synthesis call."
