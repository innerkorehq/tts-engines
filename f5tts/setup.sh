#!/usr/bin/env bash
# Sets up the isolated venv for the f5tts engine. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.13
uv sync

echo "f5tts venv ready: tts-engines/f5tts/.venv"
