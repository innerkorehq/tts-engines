"""
F5-TTS persistent synthesis daemon.

The original `scripts/f5tts_infer.py` wrapper spawns a *fresh* subprocess per
scene — meaning the ~1.3GB DiT model + vocoder get reloaded onto MPS/CUDA from
scratch on every single narration clip. For a render with N scenes that's N
full model loads instead of one.

This script instead loads the model + vocoder ONCE and then serves synthesis
requests over a simple line-delimited JSON protocol on stdin/stdout, kept
alive for the lifetime of the worker process. `TTSAdapter._synthesise_f5tts`
talks to it as a long-running daemon (spawning it lazily on first use and
auto-restarting it if it crashes), instead of shelling out per call.

Protocol (one JSON object per line, UTF-8, newline-terminated):
    Request  -> {"text": "...", "ref_audio": "...", "ref_text": "...", "output_path": "..."}
    Response <- {"ok": true}
             <- {"ok": false, "error": "..."}
A "READY" line is written to stdout once the model has finished loading, so
the parent process knows when it's safe to start sending requests.

Why a separate process at all (same reasons as the old wrapper):
  - F5-TTS's `infer_batch_process` uses a `ThreadPoolExecutor` that crashes on
    macOS MPS when batches run concurrently — patched to max_workers=1 below.
  - The venv's PYTHONPATH/VIRTUAL_ENV poison anaconda's torch dylib
    resolution, so this must run under a clean anaconda interpreter.
Both constraints are equally well satisfied by a long-lived process as by a
short-lived one — there's no need to pay the reload cost every time.
"""
import concurrent.futures as _cf

_OrigTPE = _cf.ThreadPoolExecutor


class _SequentialTPE(_OrigTPE):
    """Drop-in replacement that processes tasks one at a time (MPS isn't
    thread-safe for concurrent model_obj.sample() calls)."""

    def __init__(self, *args, **kwargs):
        kwargs["max_workers"] = 1
        super().__init__(*args, **kwargs)


_cf.ThreadPoolExecutor = _SequentialTPE  # patch before f5_tts imports

import json
import os
import sys
import traceback

# ── Protect the protocol stream from library noise ───────────────────────────
# F5-TTS's own code calls bare `print(...)` during model loading AND during
# inference (e.g. "Generating audio in N batches...", "gen_text 0 ...",
# "vocab : ...") — all of which would land on stdout and corrupt our
# line-delimited JSON protocol if left alone. So: duplicate the *original*
# stdout fd into a dedicated file object reserved exclusively for protocol
# messages (READY / JSON responses), then repoint `sys.stdout` at stderr so
# every incidental `print()` from imported libraries is harmless diagnostic
# noise instead of protocol corruption.
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT
import soundfile as sf


def _log(msg: str) -> None:
    """Diagnostic logging to stderr — stdout is reserved for the protocol."""
    print(f"[f5tts_server] {msg}", file=sys.stderr, flush=True)


# ── Memoise reference-clip preprocessing ─────────────────────────────────────
# `preprocess_ref_audio_text` loads the reference clip, runs pydub
# silence-detection/clipping, re-exports a trimmed temp wav, and md5-hashes
# it — purely to derive the *same* processed reference clip + text every time
# for a given voice profile. Cache it per (ref_audio, ref_text) so this work
# happens once per voice, not once per scene.
_ref_preprocess_cache: dict = {}


def _cached_preprocess_ref_audio_text(ref_audio_orig, ref_text, *args, **kwargs):
    key = (ref_audio_orig, ref_text)
    cached = _ref_preprocess_cache.get(key)
    if cached is None:
        cached = preprocess_ref_audio_text(ref_audio_orig, ref_text, *args, **kwargs)
        _ref_preprocess_cache[key] = cached
        _log(f"cached preprocessed reference clip for ref_audio={ref_audio_orig}")
    return cached


def _load_models():
    # Heads-up: this prints "READY-pending" status purely to stderr; the
    # actual READY signal (on the protocol stream) is sent from main() once
    # this returns.
    _log("Loading F5TTS_v1_Base model + vocoder (one-time, this may take a while)…")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
    vocoder = load_vocoder(vocoder_name="vocos")
    model_obj = load_model(DiT, model_cfg, ckpt_path, mel_spec_type="vocos", vocab_file="")
    _log("Model + vocoder ready.")
    return model_obj, vocoder


def _handle_request(req: dict, model_obj, vocoder) -> dict:
    text = req["text"]
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    output_path = req["output_path"]
    remove_silence = req.get("remove_silence", True)
    # Speed/quality knobs (see Settings.TTS_F5_NFE_STEP/CFG_STRENGTH in
    # src/config/settings.py for the trade-off explanation). Defaults below
    # match upstream F5-TTS defaults — i.e. behave exactly as before if the
    # caller doesn't override them.
    nfe_step = req.get("nfe_step", 32)
    cfg_strength = req.get("cfg_strength", 2.0)

    processed_ref_audio, processed_ref_text = _cached_preprocess_ref_audio_text(ref_audio, ref_text)

    final_wave, final_sample_rate, _spectrogram = infer_process(
        processed_ref_audio,
        processed_ref_text,
        text,
        model_obj,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
    )

    sf.write(output_path, final_wave, final_sample_rate)
    if remove_silence:
        remove_silence_for_generated_wav(output_path)

    return {"ok": True}


def main() -> int:
    model_obj, vocoder = _load_models()

    # Signal readiness to the parent process on the protected protocol stream
    # (NOT via `print`, which now points at stderr).
    _send("READY")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "SHUTDOWN":
            _log("Received SHUTDOWN — exiting.")
            return 0
        try:
            req = json.loads(line)
            resp = _handle_request(req, model_obj, vocoder)
        except Exception as exc:  # noqa: BLE001 — must always answer the request
            _log(f"Error handling request: {exc}\n{traceback.format_exc()}")
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(json.dumps(resp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
