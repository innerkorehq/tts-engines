#!/usr/bin/env bash
# Sets up the isolated venv for the hinglish-tts engine — literal vendoring
# of harrrshall/hinglish-tts's own install instructions (its README steps
# 2 and 5), adapted only where the literal commands are provably broken in
# this environment (see notes below each deviation). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11

# README step 2, run via pip (not uv sync) to match upstream's own command
# shape exactly. `numpy>=2.0,<2.1` is dropped: it directly conflicts with
# accelerate==0.33.0's own numpy<2.0 requirement (verified via pip's
# resolver — mutually unsatisfiable), so numpy is left to resolve
# automatically to whatever satisfies both accelerate and transformers
# (currently 1.26.4).
.venv/bin/python -m pip install \
    git+https://github.com/AI4Bharat/IndicF5.git \
    "transformers==4.49.0" "accelerate==0.33.0" \
    soundfile

# Not in the README (torch/torchaudio aren't pinned there, so pip resolves
# whatever's newest-compatible) — the torchaudio version this pulls in
# (2.11.0) requires torchcodec for torchaudio.load(), which ref-audio
# loading in f5_tts/infer/utils_infer.py calls directly. Without it,
# synthesize() fails immediately on every call with ImportError.
.venv/bin/python -m pip install torchcodec

# README step 5 — preprocessing (skip only for Devanagari-only input).
# Plain `pip install ai4bharat-transliteration` pulls fairseq==0.12.1 (the
# newest resolvable version), whose PyPI sdist fails to build
# (FileNotFoundError: fairseq/version.txt — a known upstream fairseq
# packaging bug, unrelated to this repo). fairseq==0.12.2's sdist builds
# cleanly. Full resolution still fails even pinned (pip rejects the only
# available omegaconf<2.1 wheels for "invalid metadata"), so this list is
# installed with --no-deps — same curated set as tts-engines/indic-xlit's
# setup.sh, which already vendors the same ai4bharat-transliteration engine.
.venv/bin/python -m pip install --no-deps \
    ai4bharat-transliteration==1.1.3 fairseq==0.12.2 ujson sacremoses \
    indic-nlp-library urduhack==1.1.1 pydload mock tensorboardx \
    progressbar2 python-utils bitarray sacrebleu portalocker tabulate \
    colorama regex lxml

# urduhack/__init__.py unconditionally imports .pipeline/.conll/
# .utils.resources, which transitively pull in TensorFlow via its NER model.
# Those modules are only used for lang_code=='ur' (never hit here — we only
# use lang='hi'). Patch __init__.py to keep just normalize/__version__/get_info.
URDUHACK_INIT=".venv/lib/python3.11/site-packages/urduhack/__init__.py"
if [ -f "$URDUHACK_INIT" ] && grep -q "from .pipeline" "$URDUHACK_INIT"; then
    cat > "$URDUHACK_INIT" <<'EOF'
# coding: utf8
"""Project Entry point"""
from .about import __version__, get_info
from .normalization import normalize

# .conll / .pipeline / .utils.resources (Pipeline, CoNLL, download) pull in
# tensorflow via the NER/POS-tagger models. ai4bharat-transliteration only
# needs `normalize` (used for Urdu/Shahmukhi normalization), so those heavy
# imports are dropped here.
__all__ = ["__version__", "get_info", "normalize"]
EOF
    echo "Patched urduhack/__init__.py to drop TensorFlow-pulling imports."
fi

.venv/bin/python -m pip install "fastapi>=0.115" "uvicorn[standard]>=0.30"

echo "hinglish-tts venv ready: tts-engines/hinglish-tts/.venv"
echo "Set HF_TOKEN in the environment (or .env) before first use — ai4bharat/IndicF5 is gated:"
echo "  https://huggingface.co/ai4bharat/IndicF5"
