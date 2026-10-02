# tts-engines

A collection of independent, self-contained text-to-speech engines, each
its own HTTP service with its own isolated dependencies (several have
mutually incompatible requirements — that's why they're separate, not a
historical accident).

Every engine exposes the same shape of API:
- `GET /health` — readiness check.
- `POST /synthesize` — synthesize speech. If the request includes an `out`
  path, the service writes the WAV file there (for callers that share a
  filesystem/volume with the service) and responds with a small JSON status
  object. If `out` is omitted, the response body *is* the WAV audio
  (`Content-Type: audio/wav`) — for any caller, anywhere.
- Errors are a normal HTTP error status with a JSON `{"detail": "..."}` body.

See each engine's own `README.md` for its exact request schema.

## Engine catalog

| Engine | Model | Notes |
|---|---|---|
| [kokoro](kokoro/) | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Fast, open, supports inline eSpeak phoneme codes for pronunciation control. |
| [f5tts](f5tts/) | [F5-TTS](https://github.com/SWivid/F5-TTS) (`F5TTS_v1_Base`) | Zero-shot voice cloning. |
| [f5tts-hinglish](f5tts-hinglish/) | [rajputsw/F5-TTS-Hinglish](https://huggingface.co/rajputsw/F5-TTS-Hinglish) | F5-TTS fine-tuned for Hindi-English code-switched speech. |
| [hinglish-tts](hinglish-tts/) | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | Gated model — requires an `HF_TOKEN`. |
| [indicf5](indicf5/) | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | Gated model — requires an `HF_TOKEN`. |
| [indic-xlit](indic-xlit/) | [IndicXlit](https://github.com/AI4Bharat/IndicXlit) | *Not audio* — bidirectional Roman<->Indic-script transliteration (used alongside the audio engines, e.g. to romanize non-Latin narration before synthesis). |
| [chatterbox-multilingual-hi](chatterbox-multilingual-hi/) | [chatterbox-multilingual-hi-mlx-fp16](https://huggingface.co/gagan1985/chatterbox-multilingual-hi-mlx-fp16) | Hindi voice cloning. **Apple Silicon only** (MLX). Caches expensive speaker-conditioning per voice on disk. |
| [mlx-audio](mlx-audio/) | 6 models: Qwen3-TTS, Chatterbox, Chatterbox-Multilingual, Voxtral-TTS, Higgs-TTS, Svara-TTS | **Apple Silicon only** (MLX). One process, at most one model resident at a time (switches on demand). |
| [omnivoice](omnivoice/) | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | 600+ languages, zero-shot cloning. Automatic MPS→CPU fallback on OOM. |
| [supertonic](supertonic/) | [Supertonic TTS 3](https://github.com/supertone-inc/supertonic) | ONNX/CPU-only — runs anywhere (the only engine here with no GPU/MPS/MLX dependency). 10 preset voices, 31 languages. |

## Running an engine

Each engine directory is independent:

```bash
cd supertonic
uv venv --python 3.13 && uv sync   # or: bash setup.sh
uv run server.py
```

**Python versions:** engines target Python 3.13, except `kokoro`,
`indic-xlit` and `hinglish-tts`, which stay on 3.11 — `kokoro`'s
`curated-tokenizers` (via misaki/spacy-curated-transformers) is source-only
and its Cython build fails on 3.13, and `indic-xlit`/`hinglish-tts` depend on
fairseq 0.12.2, whose `hydra-core`/`antlr4` pins import `typing.io` (removed in
3.13). Each engine's own `setup.sh`/`pyproject.toml` is authoritative.

Or with Docker (where supported — MLX-based engines can't run in a Linux
container; see their individual READMEs):

```bash
cd kokoro
docker build -t kokoro-tts .
docker run -p 8001:8001 kokoro-tts
```

Or bring up several at once with docker-compose (from the repo root):

```bash
docker compose up kokoro f5tts supertonic
```

## Ports

Each engine defaults to its own port (overridable via the `PORT` env var):

| Engine | Port |
|---|---|
| kokoro | 8001 |
| chatterbox-multilingual-hi | 8002 |
| f5tts | 8003 |
| f5tts-hinglish | 8004 |
| hinglish-tts | 8005 |
| indic-xlit | 8006 |
| mlx-audio | 8007 |
| omnivoice | 8008 |
| supertonic | 8009 |
| indicf5 | 8011 |

## License

Each engine wraps a separate upstream model with its own license — see the
linked model cards/repositories in the table above, and each engine's own
README.
