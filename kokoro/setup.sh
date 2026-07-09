#!/usr/bin/env bash
# Sets up the isolated venv for the kokoro engine. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "kokoro venv ready: tts-engines/kokoro/.venv"
