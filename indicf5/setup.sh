#!/usr/bin/env bash
# Sets up the isolated venv for the indicf5 engine. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "indicf5 venv ready: tts-engines/indicf5/.venv"
