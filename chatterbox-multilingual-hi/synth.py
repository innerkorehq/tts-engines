#!/usr/bin/env python
"""
Chatterbox-Multilingual-hi (ResembleAI/Chatterbox-Multilingual-hi) voice-clone
synthesis CLI.

ResembleAI/Chatterbox-Multilingual-hi is a dedicated Hindi finetune from the
"Chatterbox Multilingual V3 Single Language Pack" series — a T3 checkpoint
(t3_hi.safetensors) finetuned for Hindi, built on the same T3 architecture
(T3Config.multilingual(), 2454-token multilingual vocabulary — per the model
card) every Chatterbox multilingual checkpoint shares, plus its own copy of
a v3 speech decoder (s3gen_v3.pt) for a fully self-contained single-language
pack.

This engine loads the shared base multilingual model (ResembleAI/chatterbox
— ve/s3gen/tokenizer/conds every language shares) via
ChatterboxMultilingualTTS.from_pretrained(device=..., t3_model="v3") — the
`t3_model` kwarg only exists on chatterbox-tts's GitHub master (unreleased to
PyPI as of this writing; pinned via a git dependency in pyproject.toml, not
plain `chatterbox-tts` from PyPI) — then swaps ONLY the T3 module's weights
for the Hindi finetune — the same "download the finetuned checkpoint,
load_state_dict() onto the base engine's text-to-token module, leave the rest
of the pipeline as-is" pattern the old chatterbox-hinglish engine used for
its own T3 finetune (ketav/chatterbox-turbo-hinglish onto ChatterboxTurboTTS).
Deliberately does NOT also swap in the finetune repo's own s3gen_v3.pt —
verified against chatterbox-tts's own source that from_local() always loads
a single "s3gen.pt" regardless of which T3 version is selected (the T3/s3gen
pairing doesn't vary by version at all in this library), so the base model's
own s3gen.pt IS already the correct, only pairing — a second download and a
second, less battle-tested state-dict swap would buy nothing.

Request (stdin JSON):
    {"text": "...", "ref_audio": "/path/to/ref.wav", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": N, "duration_s": F}
"""
import logging
import os
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("chatterbox-multilingual-hi")

_BASE_REPO = "ResembleAI/chatterbox"
_FINETUNE_REPO = "ResembleAI/Chatterbox-Multilingual-hi"
_FINETUNE_CKPT = "t3_hi.safetensors"
_LANGUAGE_ID = "hi"


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or any of its dependents,
    e.g. chatterbox.mtl_tts below) before HF_HUB_OFFLINE is set makes the
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
    # locally. ChatterboxMultilingualTTS.from_pretrained() takes no
    # local_files_only kwarg, so it can't be threaded through directly —
    # HF_HUB_OFFLINE forces the same behavior transparently through any
    # huggingface_hub call, including ones buried in third-party code we
    # don't control. MUST be set before importing huggingface_hub (or
    # anything that imports it, like chatterbox below) — it's read into a
    # module-level constant at import time, not rechecked per call, so
    # setting it after the import is too late.
    if _repo_looks_cached(_BASE_REPO) and _repo_looks_cached(_FINETUNE_REPO):
        os.environ["HF_HUB_OFFLINE"] = "1"

    # mtl_tts.from_pretrained() passes token=os.getenv("HF_TOKEN") straight
    # into snapshot_download() with no empty-string guard — vidgen's own
    # .env ships HF_TOKEN= (present but empty, meant as "unset" for
    # unauthenticated public-repo access), which huggingface_hub then
    # renders as an `Authorization: Bearer ` header (trailing space, no
    # token) that httpx rejects outright with LocalProtocolError before any
    # request is even sent. Popping an empty HF_TOKEN here makes
    # os.getenv("HF_TOKEN") return None instead, which huggingface_hub
    # correctly treats as "no auth" and omits the header entirely.
    if not os.environ.get("HF_TOKEN"):
        os.environ.pop("HF_TOKEN", None)

    import torch
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    with quiet_stdout():
        logger.info("Loading Chatterbox-Multilingual-hi (base=%s, finetune=%s)…", _BASE_REPO, _FINETUNE_REPO)
        engine = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        ckpt_path = hf_hub_download(repo_id=_FINETUNE_REPO, filename=_FINETUNE_CKPT)
        state_dict = load_file(ckpt_path, device=device)
        engine.t3.load_state_dict(state_dict)
        engine.t3.to(device)
        engine.t3.eval()
        logger.info("Chatterbox-Multilingual-hi model ready on device=%s", device)

        wav = engine.generate(
            text,
            language_id=_LANGUAGE_ID,
            audio_prompt_path=ref_audio,
            temperature=0.5,
        )
        wav_np = wav.squeeze().detach().cpu().numpy() if hasattr(wav, "detach") else wav
        sf.write(out, wav_np, engine.sr)
    succeed(sample_rate=engine.sr, duration_s=len(wav_np) / engine.sr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Chatterbox-Multilingual-hi synthesis failed")
        fail(str(e))
