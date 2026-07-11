#!/usr/bin/env python
"""
maya-research/Veena (base model, GGUF-quantized by tensorblock) TTS CLI.

Llama-3-architecture (~3B param) autoregressive model that generates SNAC
(24kHz) audio-codec tokens from text; run here via llama.cpp (GGUF, CPU/Metal-
friendly) for the language-model half, then decoded to a waveform via the
`snac` package (PyTorch) for the vocoder half — same two-stage design as
Orpheus-TTS. Same architecture/control tokens/4 speakers as the
tts-engines/veena-hinglish/ sibling engine, but this is the BASE model:
native Hindi (Devanagari script) + English, not the Roman-script
Hinglish-only fine-tune.
  https://huggingface.co/shinshekai/Veena-Q8_0-GGUF  (this quantization)
  https://huggingface.co/maya-research/Veena          (base model, prompt
                                                         format, control
                                                         token IDs)

Quant choice matters a lot here, not just for audio fidelity: at the
initially-used tensorblock Q2_K/Q3_K_M quants (the only ones tensorblock
actually uploaded, despite its README advertising more), the model
frequently failed to hold the requested speaker's identity — e.g. asking
for "kavya" (female) would come out male-pitched more often than not
(median F0 ~130Hz vs. her correct ~220Hz range) across repeated runs on
identical input, confirmed via librosa pyin pitch tracking. Lowering
sampling temperature made this WORSE (more confidently wrong), which
pointed at quantization noise corrupting the speaker-conditioning signal
itself rather than a sampling-variance issue. Switching to this repo's
single clean Q8_0 conversion fixed it (correct speaker in ~4/5 repeated
runs vs. ~1/5 before) — some run-to-run variance is still expected from an
autoregressive model, but it no longer defaults to the wrong gender.

Request (stdin JSON):
    {"text": "...", "speaker": "kavya", "out": "/abs/out.wav"}
        speaker: one of kavya | agastya | maitri | vinaya (default: kavya) —
        Veena does not do reference-audio voice cloning; it selects one of 4
        fixed trained voices via a special prompt token.

Response (stdout JSON): {"ok": true, "sample_rate": 24000, "duration_s": F}
                         or {"ok": false, "error": "..."}
"""
import logging
import sys
from pathlib import Path

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("veena")

_HERE = Path(__file__).resolve().parent
_MODELS_DIR = _HERE / "models"

_GGUF_REPO = "shinshekai/Veena-Q8_0-GGUF"
_GGUF_FILENAME = "veena-q8_0.gguf"

_SNAC_REPO = "hubertsiuzdak/snac_24khz"
_SAMPLE_RATE = 24000
_N_CTX = 2048  # matches Veena's trained context length

# Control-token IDs fixed by Veena's training (maya-research/Veena README) —
# stable across quantizations/fine-tunes since they're part of the trained
# vocabulary, not something llama.cpp's tokenizer infers at runtime. Same
# values as the veena-hinglish sibling engine (same base architecture).
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266
_SNAC_CODES_PER_FRAME = 7
_SNAC_CODEBOOK_SIZE = 4096

SPEAKERS = {"kavya", "agastya", "maitri", "vinaya"}
DEFAULT_SPEAKER = "kavya"

_llama = None
_snac = None


def _get_llama():
    global _llama
    if _llama is not None:
        return _llama

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gguf_path = _MODELS_DIR / _GGUF_FILENAME
    if not gguf_path.exists():
        logger.info("Downloading %s from %s (one-time)…", _GGUF_FILENAME, _GGUF_REPO)
        downloaded = hf_hub_download(
            repo_id=_GGUF_REPO, filename=_GGUF_FILENAME, local_dir=str(_MODELS_DIR),
        )
        gguf_path = Path(downloaded)

    logger.info("Loading llama.cpp model %s…", gguf_path.name)
    _llama = Llama(model_path=str(gguf_path), n_ctx=_N_CTX, n_gpu_layers=-1, verbose=False)
    logger.info("Veena GGUF model ready (n_vocab=%d).", _llama.n_vocab())
    return _llama


def _get_snac():
    global _snac
    if _snac is not None:
        return _snac

    import torch
    from snac import SNAC

    logger.info("Loading SNAC decoder (%s)…", _SNAC_REPO)
    # Every call to this engine runs in a brand-new subprocess (see
    # TTSAdapter._run_engine_subprocess) — there's no in-process cache that
    # survives between calls, so without local_files_only, EVERY synthesis
    # call re-resolves SNAC's checkpoint against huggingface.co over the
    # network even though it's already cached locally. Falls back to a
    # normal (network-enabled) load only on the very first run.
    try:
        model = SNAC.from_pretrained(_SNAC_REPO, local_files_only=True).eval()
    except Exception:
        logger.info("Not fully cached locally yet — downloading %s…", _SNAC_REPO)
        model = SNAC.from_pretrained(_SNAC_REPO).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _snac = model.to(device)
    logger.info("SNAC decoder ready (device=%s).", device)
    return _snac


def _decode_snac_tokens(snac_tokens: list[int]):
    """De-interleave 7-tokens-per-frame SNAC codes into 3 hierarchical levels and decode."""
    import torch

    if not snac_tokens or len(snac_tokens) % _SNAC_CODES_PER_FRAME != 0:
        raise ValueError(
            f"Expected a non-zero multiple of {_SNAC_CODES_PER_FRAME} SNAC tokens, "
            f"got {len(snac_tokens)}"
        )

    snac_model = _get_snac()
    device = next(snac_model.parameters()).device
    llm_codebook_offsets = [
        AUDIO_CODE_BASE_OFFSET + i * _SNAC_CODEBOOK_SIZE for i in range(_SNAC_CODES_PER_FRAME)
    ]

    codes_lvl: list[list[int]] = [[], [], []]
    for i in range(0, len(snac_tokens), _SNAC_CODES_PER_FRAME):
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        codes_lvl[1].append(snac_tokens[i + 1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i + 4] - llm_codebook_offsets[4])
        codes_lvl[2].append(snac_tokens[i + 2] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i + 3] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i + 5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i + 6] - llm_codebook_offsets[6])

    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor >= _SNAC_CODEBOOK_SIZE)):
            raise ValueError(f"Invalid SNAC token value outside [0, {_SNAC_CODEBOOK_SIZE - 1}]")
        hierarchical_codes.append(tensor)

    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)
    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()


def synthesise(text: str, speaker: str, out_path: str) -> tuple[int, float]:
    import soundfile as sf

    if speaker not in SPEAKERS:
        logger.warning("Unknown speaker %r, falling back to %r", speaker, DEFAULT_SPEAKER)
        speaker = DEFAULT_SPEAKER

    llama = _get_llama()

    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = llama.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN,
    ]

    # Upstream Veena's example caps generation at a fixed 700 tokens (a
    # ~7-tokens-per-character heuristic calibrated for short demo sentences).
    # That cap is lower than what a single real narration sentence/scene
    # needs — anything past ~90 characters already exceeds it — so generation
    # was hitting `len(generated) >= max_tokens` and cutting off mid-utterance
    # *before* the model emitted its own END_OF_SPEECH/END_OF_AI token. Bound
    # by the actual remaining context budget instead: the model is expected to
    # self-terminate via EOS for any normal-length narration; this cap exists
    # only to prevent a runaway generation from exceeding n_ctx.
    max_tokens = max(_N_CTX - len(input_tokens) - 16, 64)

    generated: list[int] = []
    hit_token_cap = False
    for token_id in llama.generate(
        input_tokens,
        temp=0.4,
        top_p=0.9,
        repeat_penalty=1.05,
    ):
        if token_id in (END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN):
            break
        generated.append(token_id)
        if len(generated) >= max_tokens:
            hit_token_cap = True
            break

    if hit_token_cap:
        logger.warning(
            "Generation hit the %d-token cap before a natural end-of-speech token — "
            "output audio is likely truncated mid-utterance.", max_tokens,
        )

    snac_tokens = [
        t for t in generated
        if AUDIO_CODE_BASE_OFFSET <= t < (AUDIO_CODE_BASE_OFFSET + _SNAC_CODES_PER_FRAME * _SNAC_CODEBOOK_SIZE)
    ]
    if not snac_tokens:
        raise RuntimeError("No audio tokens generated")
    # Drop a trailing partial frame (can't be decoded) rather than failing outright.
    complete_len = len(snac_tokens) - (len(snac_tokens) % _SNAC_CODES_PER_FRAME)
    if complete_len == 0:
        raise RuntimeError(f"Only {len(snac_tokens)} audio token(s) generated — not enough for one frame")
    snac_tokens = snac_tokens[:complete_len]

    audio = _decode_snac_tokens(snac_tokens)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, _SAMPLE_RATE)
    duration_s = len(audio) / _SAMPLE_RATE
    return _SAMPLE_RATE, duration_s


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    speaker = req.get("speaker", DEFAULT_SPEAKER)
    out_path = req.get("out")

    if not text.strip():
        fail("text is empty")
        return
    if not out_path:
        fail("out path is required")
        return

    try:
        with quiet_stdout():
            sample_rate, duration_s = synthesise(text, speaker, out_path)
    except Exception as e:
        logger.exception("veena synthesis failed")
        fail(str(e))
        return

    succeed(sample_rate=sample_rate, duration_s=duration_s)


if __name__ == "__main__":
    main()
