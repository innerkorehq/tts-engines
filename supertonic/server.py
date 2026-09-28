#!/usr/bin/env python
"""
Supertonic TTS persistent synthesis daemon.

The original `synth.py` CLI is invoked as a *fresh* subprocess per scene —
meaning the ONNX model gets reloaded from disk on every single narration
clip. For a render with N scenes that's N full model loads instead of one.

This script instead loads the Supertonic ONNX model ONCE and then serves
synthesis requests over a simple line-delimited JSON protocol on
stdin/stdout, kept alive for the lifetime of the worker process. The caller
(`TTSAdapter`) talks to it as a long-running daemon (spawning it lazily on
first use and auto-restarting it if it crashes / falling back to the
one-shot `synth.py` if it fails to start), instead of shelling out per call.

Protocol (one JSON object per line, UTF-8, newline-terminated):
    Request  -> {"text": "...", "out": "...", "voice": "M1", "lang": "en",
                 "speed": 1.05, "steps": 8}
    Response <- {"ok": true, "sample_rate": 44100, "duration_s": F}
             <- {"ok": false, "error": "..."}
A "READY" line is written to the protocol stream once the model has
finished loading, so the parent process knows when it's safe to start
sending requests. A "SHUTDOWN" line (exact string, not JSON) exits cleanly.
"""
import json
import os
import sys
import traceback

# ── Protect the protocol stream from library noise ───────────────────────────
# Supertonic's own pipeline code calls bare `print(...)` in a few spots (e.g.
# per-chunk progress, "Generation complete!"), though only when `verbose=True`
# is passed to `synthesize()` — which this daemon never does, matching
# synth.py's behaviour. onnxruntime itself may also print provider/warning
# noise during session creation. Since none of this is guaranteed to stay
# silent across versions, apply the same defensive stdout-hijack as
# f5tts_server.py regardless: duplicate the *original* stdout fd into a
# dedicated file object reserved exclusively for protocol messages (READY /
# JSON responses), then repoint `sys.stdout` at stderr so any incidental
# `print()` from imported libraries is harmless diagnostic noise instead of
# protocol corruption. This must happen BEFORE importing any ML libraries
# (soundfile, supertonic/onnxruntime).
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _send(line: str) -> None:
    _protocol_out.write(line + "\n")
    _protocol_out.flush()


import soundfile as sf

_SAMPLE_RATE = 44_100


def _log(msg: str) -> None:
    """Diagnostic logging to stderr — stdout is reserved for the protocol."""
    print(f"[supertonic_server] {msg}", file=sys.stderr, flush=True)


def _load_model():
    # One-time setup: import + construct the TTS engine (loads the ONNX
    # session(s) from disk, downloading model files first if needed — same
    # as `_get_tts()` in synth.py, just run unconditionally at startup
    # instead of lazily on first request.
    _log("Loading Supertonic TTS 3 (one-time, model download may occur)…")
    from supertonic import TTS
    tts = TTS(auto_download=True)
    _log("Supertonic TTS 3 ready.")
    return tts


def _handle_request(req: dict, tts) -> dict:
    text = req.get("text", "")
    out = req.get("out", "")
    voice = req.get("voice") or "M1"
    lang = req.get("lang") or "na"
    speed = float(req.get("speed") or 1.05)
    steps = int(req.get("steps") or 8)

    if not text.strip():
        return {"ok": False, "error": "text is required"}
    if not out:
        return {"ok": False, "error": "out path is required"}

    # Per-request work: load the requested voice style and run inference.
    # `get_voice_style` re-reads the style file per call (no shared mutable
    # cache to worry about); `synthesize` only touches `tts.model`, the
    # already-loaded ONNX session — safe to reuse across many calls, which
    # is the whole point of loading it once.
    style = tts.get_voice_style(voice_name=voice)
    wav, duration = tts.synthesize(
        text=text,
        lang=lang,
        voice_style=style,
        total_steps=steps,
        speed=speed,
    )
    # wav: float32 numpy array (1, num_samples); trim to actual duration
    samples = wav[0, : int(_SAMPLE_RATE * duration[0].item())]
    sf.write(out, samples, _SAMPLE_RATE)
    duration_s = float(len(samples) / _SAMPLE_RATE)
    _log(f"Synthesized {duration_s:.2f}s → {out}")

    return {"ok": True, "sample_rate": _SAMPLE_RATE, "duration_s": duration_s}


def main() -> int:
    tts = _load_model()

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
            resp = _handle_request(req, tts)
        except Exception as exc:  # noqa: BLE001 — must always answer the request
            _log(f"Error handling request: {exc}\n{traceback.format_exc()}")
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(json.dumps(resp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
