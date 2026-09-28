"""
IndicF5 (ai4bharat/IndicF5) persistent synthesis daemon.

The original `synth.py` CLI is invoked as a *fresh* subprocess per scene —
meaning the IndicF5 model (which bundles its own vocoder, loaded via
`transformers.AutoModel(..., trust_remote_code=True)`) gets reloaded from
scratch on every single narration clip. For a render with N scenes that's N
full model loads instead of one.

This script instead loads the model ONCE and then serves synthesis requests
over a simple line-delimited JSON protocol on stdin/stdout, kept alive for
the lifetime of the worker process — mirroring
`tts-engines/f5tts/f5tts_server.py`. The caller (`TTSAdapter`) talks to it as
a long-running daemon (spawning it lazily on first use and auto-restarting
it if it crashes), instead of shelling out to `synth.py` per call. `synth.py`
itself is left untouched and still works as a one-shot fallback if the
daemon can't be started.

Protocol (one JSON object per line, UTF-8, newline-terminated):
    Request  -> {"text": "...", "ref_audio": "...", "ref_text": "...", "output_path": "..."}
    Response <- {"ok": true, "sample_rate": 24000, "duration_s": F}
             <- {"ok": false, "error": "..."}
A "READY" line is written to the protocol stream once the model has
finished loading, so the parent process knows when it's safe to start
sending requests. A bare "SHUTDOWN" line exits the process cleanly.
"""
import glob
import os
import shutil
import sys

# ── Protect the protocol stream from library noise ───────────────────────────
# IndicF5 (and the transformers/f5_tts machinery it pulls in via
# trust_remote_code) print progress/status directly via bare `print(...)`
# during model loading AND during inference — all of which would land on
# stdout and corrupt our line-delimited JSON protocol if left alone. So:
# duplicate the *original* stdout fd into a dedicated file object reserved
# exclusively for protocol messages (READY / JSON responses), then repoint
# `sys.stdout` at stderr so every incidental `print()` from imported
# libraries is harmless diagnostic noise instead of protocol corruption.
# This must happen BEFORE any ML-library imports below.
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


import json
import logging
import traceback

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("indicf5_server")


def _log(msg: str) -> None:
    """Diagnostic logging to stderr — stdout is reserved for the protocol."""
    print(f"[indicf5_server] {msg}", file=sys.stderr, flush=True)


def _patch_indicf5_for_mps() -> None:
    """
    IndicF5 ships its model code via HF `trust_remote_code` (downloaded to
    ~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/<rev>/model.py).
    That file hardcodes:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    — i.e. it never considers Apple Silicon's MPS backend, so on macOS it
    always falls back to CPU (slow). Patch the cached file in-place to add an
    MPS branch. Idempotent (checks a marker before writing) and self-heals if
    HF re-downloads a fresh copy of the file.
    """
    try:
        import torch
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return  # nothing to do on non-Apple-Silicon machines
    except Exception:
        return

    cache_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5")
    marker = "torch.backends.mps.is_available()"
    old = 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
    new = (
        'device = torch.device(\n'
        '            "cuda" if torch.cuda.is_available()\n'
        '            else "mps" if torch.backends.mps.is_available()\n'
        '            else "cpu"\n'
        '        )'
    )

    for path in glob.glob(os.path.join(cache_root, "*", "model.py")):
        try:
            src = open(path, "r", encoding="utf-8").read()
            if marker in src:
                continue  # already patched
            if old not in src:
                logger.warning("IndicF5 model.py at %s has unexpected device-selection code — skipping MPS patch.", path)
                continue
            patched = src.replace(old, new, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(patched)
            shutil.rmtree(os.path.join(os.path.dirname(path), "__pycache__"), ignore_errors=True)
            logger.info("Patched IndicF5 model.py for Apple Silicon MPS support: %s", path)
        except Exception as e:
            logger.warning("Could not patch IndicF5 model.py at %s for MPS: %s", path, e)


# f5_tts.infer.utils_infer.infer_process() (vendored into IndicF5's model.py
# via `from f5_tts.infer.utils_infer import ...`) computes the output audio
# canvas size from UTF-8 *byte* counts of the reference/generation texts:
#
#   ref_text_len = len(ref_text.encode("utf-8"))
#   gen_text_len = len(gen_text.encode("utf-8"))
#
# Devanagari encodes as ~3 bytes/char vs ~1 byte/char for ASCII. For
# Roman-script input this under-allocates canvas relative to a Devanagari
# generation text, truncating the synthesised audio. Counting non-whitespace
# *characters* instead fixes the ref:gen ratio regardless of script. This
# mirrors the "Mode A" duration fix from https://github.com/harrrshall/hinglish-tts.
_DURATION_ORIG_BLOCK = (
    '            # Calculate duration\n'
    '            ref_text_len = len(ref_text.encode("utf-8"))\n'
    '            gen_text_len = len(gen_text.encode("utf-8"))\n'
)
_DURATION_PATCHED_BLOCK = (
    '            # PATCHED — character-count proportional duration (Mode A fix)\n'
    '            ref_text_len = sum(1 for c in ref_text if not c.isspace())  # vidgen-patch\n'
    '            gen_text_len = sum(1 for c in gen_text if not c.isspace())\n'
)
_DURATION_SENTINEL = "vidgen-patch"


def _patch_indicf5_duration_canvas() -> None:
    """Patch the installed f5_tts package's utils_infer.py in place — see block comment above."""
    try:
        import f5_tts.infer.utils_infer as _utils_infer
    except Exception:
        logger.warning("IndicF5 duration patch: f5_tts not importable — skipping.")
        return

    path = _utils_infer.__file__
    try:
        src = open(path, "r", encoding="utf-8").read()
        if _DURATION_SENTINEL in src:
            return  # already patched
        if _DURATION_ORIG_BLOCK not in src:
            logger.warning("IndicF5 duration patch: utils_infer.py at %s has unexpected code — skipping.", path)
            return
        patched = src.replace(_DURATION_ORIG_BLOCK, _DURATION_PATCHED_BLOCK, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        shutil.rmtree(os.path.join(os.path.dirname(path), "__pycache__"), ignore_errors=True)
        logger.info("Patched f5_tts utils_infer.py for character-count duration: %s", path)
    except Exception as e:
        logger.warning("Could not patch f5_tts utils_infer.py at %s for duration: %s", path, e)


def _load_model():
    """Load ai4bharat/IndicF5 via transformers.AutoModel(trust_remote_code=True).

    One-time setup — runs once at daemon startup, before READY is signalled.
    Identical to synth.py's load_model(), just renamed to avoid colliding
    with any future module-level import of synth.py.
    """
    import torch as _torch
    import transformers.modeling_utils as _mu
    from transformers import AutoModel

    _patch_indicf5_for_mps()
    _patch_indicf5_duration_canvas()

    # transformers ALWAYS constructs `cls(config, ...)` inside a meta-device
    # context (regardless of `low_cpu_mem_usage`). IndicF5's custom __init__
    # loads its own vocoder checkpoint and calls plain `.to(device)` on it —
    # that blows up with "Cannot copy out of meta tensor; no data!" because
    # the vocoder submodules were created on the meta device. Temporarily
    # strip the meta-device context so the whole model (including the
    # vocoder loaded inside __init__) is built with real, materialised
    # tensors.
    _PTM = _mu.PreTrainedModel
    _orig_get_init_context = _PTM.get_init_context.__func__

    @classmethod
    def _get_init_context_no_meta(cls_, dtype, is_quantized, _is_ds_init_called):
        contexts = _orig_get_init_context(cls_, dtype, is_quantized, _is_ds_init_called)
        return [c for c in contexts if c != _torch.device("meta")]

    # IndicF5's custom INF5Model.__init__ calls `super().__init__(config)` but
    # never calls `self.post_init()` (its code predates the transformers
    # version that requires it). Without it, `all_tied_weights_keys` /
    # `_tp_plan` / `_no_split_modules` etc. are never set, and
    # `_finalize_model_loading` blows up. Run `post_init()` retroactively —
    # it only computes this metadata from the already-built module tree.
    _orig_finalize = _PTM._finalize_model_loading

    @staticmethod
    def _finalize_model_loading_compat(model, load_config, loading_info):
        if not hasattr(model, "all_tied_weights_keys"):
            model.post_init()
        return _orig_finalize(model, load_config, loading_info)

    _PTM.get_init_context = _get_init_context_no_meta
    _PTM._finalize_model_loading = _finalize_model_loading_compat
    try:
        logger.info("Loading IndicF5 model from ai4bharat/IndicF5…")
        # Unlike the one-shot synth.py (where every call is a brand-new
        # subprocess, so local_files_only avoided a network round-trip on
        # every synthesis call), the daemon only loads the model once for
        # its entire lifetime — but we keep the same local_files_only-first
        # strategy since it's still strictly faster/safer when the checkpoint
        # is already cached, and falls back to a network-enabled load if not.
        try:
            model = AutoModel.from_pretrained(
                "ai4bharat/IndicF5", trust_remote_code=True, low_cpu_mem_usage=False, local_files_only=True,
            )
        except Exception:
            logger.info("Not fully cached locally yet — downloading ai4bharat/IndicF5…")
            model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, low_cpu_mem_usage=False)
    finally:
        _PTM.get_init_context = classmethod(_orig_get_init_context)
        _PTM._finalize_model_loading = staticmethod(_orig_finalize)
    logger.info("IndicF5 model ready.")
    return model


def _synthesise(model, text: str, ref_audio: str, ref_text: str, out: str) -> tuple:
    """Per-request inference — identical body to synth.py's synthesise()."""
    import numpy as np
    import soundfile as sf

    audio = model(text, ref_audio_path=ref_audio, ref_text=ref_text)
    if hasattr(audio, "dtype") and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    audio = np.array(audio, dtype=np.float32)
    sample_rate = 24000
    sf.write(out, audio, samplerate=sample_rate)
    return sample_rate, len(audio) / sample_rate


def _handle_request(req: dict, model) -> dict:
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    # Accept both the f5tts-daemon convention (`output_path`) and synth.py's
    # own field name (`out`), so this daemon works with either caller shape.
    output_path = req.get("output_path") or req["out"]

    sample_rate, duration_s = _synthesise(model, text, ref_audio, ref_text, output_path)

    return {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}


def main() -> int:
    _log("Loading IndicF5 model (one-time, this may take a while)…")
    model = _load_model()
    _log("Model ready.")

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
            resp = _handle_request(req, model)
        except Exception as exc:  # noqa: BLE001 — must always answer the request
            _log(f"Error handling request: {exc}\n{traceback.format_exc()}")
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(json.dumps(resp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
