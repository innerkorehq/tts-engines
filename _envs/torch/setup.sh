#!/usr/bin/env bash
# Builds the ONE shared PyTorch venv used by f5tts, f5tts-hinglish and omnivoice.
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.13 --allow-existing
uv sync

echo "shared torch venv ready: tts-engines/_envs/torch/.venv"
