#!/usr/bin/env bash
# Sets up the isolated venv for the veena (base) engine. Safe to re-run.
#
# llama-cpp-python has no prebuilt wheel on PyPI for this version — it builds
# from source (needs cmake + a C/C++ compiler, both already required elsewhere
# in this repo's toolchain). The GGUF model weights (shinshekai's Q8_0
# conversion — see synth.py's docstring for why Q8_0 specifically, not a
# lower quant) are downloaded lazily on first real request, not here, to
# keep this script itself fast.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

echo "veena venv ready: tts-engines/veena/.venv"
echo "Model weights (~4GB, Q8_0) download on first use."
