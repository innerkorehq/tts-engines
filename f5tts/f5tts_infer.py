"""
F5-TTS synthesis wrapper — forces sequential batch processing.

F5-TTS infer_batch_process uses a ThreadPoolExecutor to run all text batches
in parallel.  On macOS (MPS / CPU) concurrent calls to model_obj.sample() are
not thread-safe and crash with 0/N batches completed.

This wrapper monkey-patches ThreadPoolExecutor to max_workers=1 before any
f5_tts import so all batches are processed sequentially.

Usage (same flags as f5-tts_infer-cli):
    python scripts/f5tts_infer.py --model F5TTS_v1_Base --ref_audio ... \\
        --ref_text "..." --gen_text "..." --output_dir ... --output_file ...
"""
import concurrent.futures as _cf

_OrigTPE = _cf.ThreadPoolExecutor


class _SequentialTPE(_OrigTPE):
    """Drop-in replacement that processes tasks one at a time."""

    def __init__(self, *args, **kwargs):
        kwargs["max_workers"] = 1
        super().__init__(*args, **kwargs)


_cf.ThreadPoolExecutor = _SequentialTPE  # patch before f5_tts imports

# Now load and run the real CLI entry point
from f5_tts.infer.infer_cli import main  # noqa: E402
import sys  # noqa: E402

sys.exit(main())
