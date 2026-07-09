#!/usr/bin/env python
"""
Indic Parler-TTS (ai4bharat/indic-parler-tts) synthesis CLI.

Description-controlled multilingual TTS (NOT voice cloning): a natural-
language `voice_description` selects one of 69 built-in speaker voices across
21 Indian languages + English and shapes pitch, speaking rate, expressivity,
recording quality, etc.

Runs on parler-tts==0.2.2's native transformers~=4.46 pin — no compat shims
required.

Request (stdin JSON):
    {"text": "...", "voice_description": "...", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": N, "duration_s": F}
"""
import logging
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("indic-parler-tts")

_INDIC_PARLER_REPO = "ai4bharat/indic-parler-tts"


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    voice_description = req.get("voice_description", "")
    out = req["out"]

    if not voice_description.strip():
        raise RuntimeError("voice_description is empty — required for engine 'indic-parler-tts'.")

    import torch
    import soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Every call to this engine runs in a brand-new subprocess (see
    # TTSAdapter._run_engine_subprocess) — there's no in-process cache that
    # survives between calls, so without local_files_only, EVERY synthesis
    # call re-resolves every config/tokenizer/checkpoint file against
    # huggingface.co over the network, even though everything is already
    # cached locally. Falls back to a normal (network-enabled) load only
    # on the very first run, when nothing is cached yet.
    def _from_pretrained_local_first(cls, repo_id, **kwargs):
        try:
            return cls.from_pretrained(repo_id, local_files_only=True, **kwargs)
        except Exception:
            logger.info("Not fully cached locally yet — downloading %s…", repo_id)
            return cls.from_pretrained(repo_id, **kwargs)

    with quiet_stdout():
        logger.info("Loading Indic Parler-TTS (%s)…", _INDIC_PARLER_REPO)
        model = _from_pretrained_local_first(ParlerTTSForConditionalGeneration, _INDIC_PARLER_REPO)

        # Move to fp16 on MPS/CUDA: in fp32, a single generate() call on this
        # ~970M-param model (with its 9 parallel DAC codebook streams through the
        # decoder) drives MPS's "driver allocated" memory to ~18GB on its very
        # first decode step, exceeding MPS's default high-watermark (~20GB on a
        # 16GB-unified-memory Mac) and raising "MPS backend out of memory".
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = model.to(device=device, dtype=dtype)
        tokenizer = _from_pretrained_local_first(AutoTokenizer, _INDIC_PARLER_REPO)
        desc_tokenizer = _from_pretrained_local_first(AutoTokenizer, model.config.text_encoder._name_or_path)
        logger.info("Indic Parler-TTS model ready on device=%s", device)

        desc_inputs = desc_tokenizer(voice_description, return_tensors="pt").to(device)
        prompt_inputs = tokenizer(text, return_tensors="pt").to(device)

        # Cap generation length: this model's decoder runs ~86 audio-codec
        # tokens/sec of output, and both the per-step compute cost AND the
        # growing KV cache scale with sequence length. The default
        # `generation_config.max_length=2610` (with `do_sample=True` and no
        # guarantee of an early EOS) could take well over an hour on MPS and blow
        # past MPS's memory ceiling well before that. 300 tokens (~3.5s of audio)
        # leaves comfortable headroom while bounding worst-case latency to
        # roughly a minute.
        #
        # `min_new_tokens` guards against the opposite failure mode: with
        # `do_sample=True`, the EOS token (1024) can occasionally be sampled
        # within the first 1-2 decode steps, producing ~0.02s of near-silent
        # audio. A floor of 50 tokens (~0.6s) avoids that degenerate output.
        with torch.inference_mode():
            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                min_new_tokens=50,
                max_new_tokens=300,
            )

        audio = generation.to(torch.float32).cpu().numpy().squeeze()
        sample_rate = model.config.sampling_rate
        sf.write(out, audio, sample_rate)
    succeed(sample_rate=sample_rate, duration_s=len(audio) / sample_rate)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Indic Parler-TTS synthesis failed")
        fail(str(e))
