#!/usr/bin/env bash
# f5tts uses the shared PyTorch venv (tts-engines/_envs/torch) instead of its own
# ~1 GB copy; this builds it if needed and points .venv at it. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

bash ../_envs/torch/setup.sh
rm -rf .venv
ln -sfn ../_envs/torch/.venv .venv

echo "f5tts venv ready: tts-engines/f5tts/.venv -> _envs/torch/.venv"
