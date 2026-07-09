#!/usr/bin/env python
"""
Chatterbox-Turbo-Hinglish (ketav/chatterbox-turbo-hinglish) voice-clone
synthesis CLI.

Runs on chatterbox-tts's native transformers==5.2.0 pin — no compat shims
required.

Request (stdin JSON):
    {"text": "...", "ref_audio": "/path/to/ref.wav", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": N, "duration_s": F}
"""
import logging
import os
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("chatterbox-hinglish")

_CHATTERBOX_HINGLISH_REPO = "ketav/chatterbox-turbo-hinglish"
_CHATTERBOX_HINGLISH_CKPT = "t3_turbo_finetuned.safetensors"


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or any of its dependents,
    e.g. chatterbox.tts_turbo below) before HF_HUB_OFFLINE is set makes the
    env var a no-op (huggingface_hub reads it into a module-level constant
    at import time, not dynamically per call).
    """
    from pathlib import Path

    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    out = req["out"]

    # Every call to this engine runs in a brand-new subprocess (see
    # TTSAdapter._run_engine_subprocess) — there's no in-process cache that
    # survives between calls, so without forcing offline mode, EVERY
    # synthesis call re-resolves every checkpoint file against
    # huggingface.co over the network, even though it's already cached
    # locally. ChatterboxTurboTTS.from_pretrained() takes no HF kwargs at
    # all (just `device`), so local_files_only can't be threaded through
    # directly — HF_HUB_OFFLINE forces the same behavior transparently
    # through any huggingface_hub call, including ones buried in
    # third-party code we don't control. MUST be set before importing
    # huggingface_hub (or anything that imports it, like chatterbox below)
    # — it's read into a module-level constant at import time, not
    # rechecked per call, so setting it after the import is too late.
    if _repo_looks_cached(_CHATTERBOX_HINGLISH_REPO) and _repo_looks_cached("ResembleAI/chatterbox-turbo"):
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    with quiet_stdout():
        logger.info("Loading Chatterbox-Turbo-Hinglish (%s)…", _CHATTERBOX_HINGLISH_REPO)
        engine = ChatterboxTurboTTS.from_pretrained(device=device)
        ckpt_path = hf_hub_download(repo_id=_CHATTERBOX_HINGLISH_REPO, filename=_CHATTERBOX_HINGLISH_CKPT)
        state_dict = load_file(ckpt_path, device=device)
        engine.t3.load_state_dict(state_dict)
        engine.t3.to(device)
        engine.t3.eval()
        logger.info("Chatterbox-Turbo-Hinglish model ready on device=%s", device)

        wav = engine.generate(text, audio_prompt_path=ref_audio, temperature=0.5)
        wav_np = wav.squeeze().detach().cpu().numpy() if hasattr(wav, "detach") else wav
        sf.write(out, wav_np, engine.sr)
    succeed(sample_rate=engine.sr, duration_s=len(wav_np) / engine.sr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Chatterbox-Turbo-Hinglish synthesis failed")
        fail(str(e))
