#!/usr/bin/env bash
# Sets up the isolated venv for the veena-hinglish engine. Safe to re-run.
#
# llama-cpp-python has no prebuilt wheel on PyPI for this version — it builds
# from source (needs cmake + a C/C++ compiler, both already required elsewhere
# in this repo's toolchain). The GGUF model weights (~2.5GB for the default
# Q4_K_M quant) are downloaded lazily on first real request, not here — see
# synth.py — to keep this script itself fast.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "veena-hinglish venv ready: tts-engines/veena-hinglish/.venv"
echo "Model weights download on first use (~2.5GB, VEENA_QUANT env var to override quant)."
