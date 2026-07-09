#!/usr/bin/env bash
# Sets up the isolated venv for the indic-xlit transliteration engine.
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.11
uv sync

# ai4bharat-transliteration's transformer backend pulls in fairseq, which in
# turn pulls in a long tail of packages (some via urduhack -> tensorflow) that
# aren't actually needed for the indic<->roman / hinglish<->devanagari paths
# used here. Install the curated set with --no-deps to avoid that bloat.
uv pip install --no-deps \
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

echo "indic-xlit venv ready: tts-engines/indic-xlit/.venv"
