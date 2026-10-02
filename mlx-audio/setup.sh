#!/usr/bin/env bash
# Sets up the isolated venv for the consolidated mlx-audio engine. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.13
uv sync

# ── Higgs-TTS BF16 patch ───────────────────────────────────────────────────
# bosonai/higgs-tts-3-4b stores ALL weights (including codec) in bfloat16.
# mlx-audio's safetensors loader (framework="mlx") can't read bfloat16 and
# its fallback (framework="pt") requires torch, which is not in this venv.
# Patch: add a pure-numpy BF16 reader and use it when torch isn't available.
HIGGS_CODEC=".venv/lib/python3.11/site-packages/mlx_audio/codec/models/higgs_audio/higgs_audio.py"
SENTINEL="# vidgen-bf16-patch"
if ! grep -q "$SENTINEL" "$HIGGS_CODEC" 2>/dev/null; then
  python3 - "$HIGGS_CODEC" <<'PATCHEOF'
import sys, re

path = sys.argv[1]
src = open(path).read()

# 1. Add helper function + sentinel after the imports block
helper = '''
# vidgen-bf16-patch
import struct as _struct

def _load_bf16_safetensors_numpy(shard_path, keys, prefix):
    """BF16 = upper 16 bits of float32. Load without torch."""
    import json as _json
    loaded = {}
    with open(shard_path, "rb") as f:
        hdr_len = _struct.unpack_from("<Q", f.read(8))[0]
        header = _json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
        scan_keys = (
            [k for k in header if k != "__metadata__" and k.startswith(prefix)]
            if keys is None else keys
        )
        for key in scan_keys:
            meta = header.get(key)
            if meta is None or meta.get("dtype") != "BF16":
                continue
            offsets = meta["data_offsets"]
            shape = meta["shape"]
            f.seek(data_start + offsets[0])
            raw = f.read(offsets[1] - offsets[0])
            import numpy as _np
            u16 = _np.frombuffer(raw, dtype=_np.uint16)
            u32 = u16.astype(_np.uint32) << 16
            arr = u32.view(_np.float32)
            if shape:
                arr = arr.reshape(shape)
            import mlx.core as _mx
            loaded[key[len(prefix):]] = _mx.array(arr)
    return loaded

'''

src = src.replace(
    "from .config import HiggsAudioConfig",
    helper + "from .config import HiggsAudioConfig",
    1,
)

# 2. Patch the fallback call site
old = ('                raw.update(_load_shard_codec_tensors(shard_path, keys, framework="pt"))')
new = (
    '                try:\n'
    '                    raw.update(_load_shard_codec_tensors(shard_path, keys, framework="pt"))\n'
    '                except (ModuleNotFoundError, ImportError):\n'
    '                    raw.update(_load_bf16_safetensors_numpy(shard_path, keys, prefix))'
)
src = src.replace(old, new, 1)

open(path, "w").write(src)
print("Higgs BF16 patch applied.")
PATCHEOF
fi

echo "mlx-audio venv ready: tts-engines/mlx-audio/.venv"
echo "Model weights download lazily on first use of each model."
