#!/usr/bin/env bash
# Sets up the isolated venv for the omnivoice engine. Safe to re-run.
# Pulls in torch+torchaudio and the official `omnivoice` PyPI package.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.13
uv sync

echo "omnivoice venv ready: tts-engines/omnivoice/.venv"
echo "Model weights download lazily from k2-fsa/OmniVoice on first use."
