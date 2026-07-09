#!/usr/bin/env bash
# Sets up the isolated venv for the indicf5-hinglish engine. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "indicf5-hinglish venv ready: tts-engines/indicf5-hinglish/.venv"
