# mlx-audio

An HTTP service consolidating 6 [mlx-audio](https://github.com/Blaizzy/mlx-audio)
TTS model variants behind one process: `qwen3-tts`, `chatterbox`,
`chatterbox-multilingual`, `voxtral-tts`, `higgs-tts`, `svara-tts`.

> **Apple Silicon only.** This engine depends on mlx-audio, which requires
> Apple's MLX framework — macOS on Apple Silicon (M-series). It will not
> install or run on Linux/CUDA/Windows.

At most **one** sub-model is kept resident in memory at a time (they range
1-4GB+ each): the first request for a given model loads it and keeps it
warm; a request for a *different* model unloads the current one first. This
trades "switching models costs a reload" for a much lower peak memory
footprint — the right trade-off when a deployment mostly reuses one
voice/model repeatedly.

## Run it

```bash
uv venv --python 3.13 && uv sync   # or: bash setup.sh
uv run server.py                    # PORT=8007 by default
```

## API

### `GET /health`

```json
{"status": "ok", "resident_model": "qwen3-tts"}
```

`resident_model` is `null` until the first `/synthesize` call for any model.

### `POST /synthesize`

Common fields:

```json
{"model": "qwen3-tts", "text": "Hello world", "out": "/path/to/out.wav"}
```

- `model` — one of `qwen3-tts`, `chatterbox`, `chatterbox-multilingual`,
  `voxtral-tts`, `higgs-tts`, `svara-tts`.
- `text` — required.
- `out` — **optional**. Only meaningful if this service shares a filesystem
  or volume with the caller. See response shapes below.

Model-specific fields:

| model | extra fields |
|---|---|
| `qwen3-tts` | `ref_audio`, `ref_text`, `language` (default `English`) |
| `chatterbox` | `ref_audio` (optional; if given, must be a clip **longer than 5s**) |
| `chatterbox-multilingual` | `ref_audio` (optional), `language` (one of `ar,da,de,el,en,es,fi,fr,he,hi,it,ja,ko,ms,nl,no,pl,pt,ru,sv,sw,tr,zh`, default `en`) |
| `voxtral-tts` | `voice` (one of e.g. `neutral_female`, `hi_male`, `fr_female`, ... — see `server.py`'s `_VOXTRAL_VOICES` for the full list). Note: `hi_*` voices expect genuine Devanagari-script text, not romanized Hindi. |
| `higgs-tts` | `ref_audio`, `ref_text` (both optional) |
| `svara-tts` | `voice` — `"<Language> (<Gender>)"`, e.g. `"Hindi (Female)"`; 19 Indic languages x 2 genders (see `server.py`'s `_SVARA_TTS_LANGUAGES`) |

**If `out` is given** (co-located mode): the service writes the WAV file to
that path and responds with:

```json
{"ok": true, "sample_rate": 24000, "duration_s": 1.23}
```

**If `out` is omitted** (standalone mode): the response body *is* the WAV
audio (`Content-Type: audio/wav`).

**On error**: an HTTP error status (422 for an unknown model/voice/language
or invalid input, 500 for a synthesis failure) with a JSON
`{"detail": "..."}` body.

## Model licenses

Each sub-model is a separate upstream project with its own license — see
the respective Hugging Face model cards linked from each repo id in
`server.py`.
