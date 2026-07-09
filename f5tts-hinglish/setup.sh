#!/usr/bin/env bash
# Sets up the isolated venv for the f5tts-hinglish engine. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "f5tts-hinglish venv ready: tts-engines/f5tts-hinglish/.venv"
