#!/usr/bin/env bash
# Sets up the isolated venv for the indic-parler-tts engine. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "indic-parler venv ready: tts-engines/indic-parler/.venv"
