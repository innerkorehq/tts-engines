#!/usr/bin/env bash
# Sets up the isolated venv for the chatterbox-multilingual-hi engine. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "chatterbox-multilingual-hi venv ready: tts-engines/chatterbox-multilingual-hi/.venv"
