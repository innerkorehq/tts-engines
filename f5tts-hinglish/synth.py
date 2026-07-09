#!/usr/bin/env python
"""
F5-TTS-Hinglish (rajputsw/F5-TTS-Hinglish) voice-clone synthesis CLI.

A fine-tune of SPRINGLab/F5-Hindi-24KHz (F5-TTS "Small" DiT config — 768-dim,
18 layers, 12 heads, vocos vocoder, 24kHz/100-mel) for Hindi-English
code-switched ("Hinglish") narration. Loaded via the stock `f5_tts.api.F5TTS`
class with the bundled "F5TTS_Small" architecture config plus this repo's own
checkpoint (`model_last.pt`) and vocabulary (`vocab.txt`), downloaded from the
Hugging Face Hub and cached locally.

Request (stdin JSON):
    {"text": "...", "ref_audio": "/path/to/ref.wav", "ref_text": "...", "out": "/path/to/out.wav"}

Response (stdout JSON): {"ok": true, "sample_rate": 24000, "duration_s": F}
"""
import logging
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("f5tts-hinglish")

_HF_REPO = "rajputsw/F5-TTS-Hinglish"
_MODEL_CONFIG = "F5TTS_Small"  # 768-dim/18-layer/12-head DiT, matches this checkpoint


def main() -> None:
    req = read_request()
    text = req.get("text", "")
    ref_audio = req["ref_audio"]
    ref_text = req["ref_text"]
    out = req["out"]

    with quiet_stdout():
        from huggingface_hub import hf_hub_download
        from f5_tts.api import F5TTS

        logger.info("Fetching %s checkpoint/vocab from Hugging Face Hub…", _HF_REPO)
        ckpt_file = hf_hub_download(repo_id=_HF_REPO, filename="model_last.pt")
        vocab_file = hf_hub_download(repo_id=_HF_REPO, filename="vocab.txt")

        logger.info("Loading F5-TTS-Hinglish (%s, config=%s)…", _HF_REPO, _MODEL_CONFIG)
        tts = F5TTS(model=_MODEL_CONFIG, ckpt_file=ckpt_file, vocab_file=vocab_file)
        logger.info("F5-TTS-Hinglish model ready on device=%s", tts.device)

        wav, sample_rate, _spec = tts.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            show_info=lambda *a, **k: None,
            remove_silence=True,
            # f5_tts.api.F5TTS.infer(seed=None) does
            # `random.randint(0, sys.maxsize)`, which is almost always out of
            # PYTHONHASHSEED's valid [0, 4294967295] range and crashes any
            # forked worker with "Fatal Python error: config_init_hash_seed".
            # Pass an in-range seed explicitly to avoid that.
            seed=0,
        )

        import soundfile as sf
        sf.write(out, wav, sample_rate)

    succeed(sample_rate=sample_rate, duration_s=len(wav) / sample_rate)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("F5-TTS-Hinglish synthesis failed")
        fail(str(e))
