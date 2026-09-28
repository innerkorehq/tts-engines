# kokoro

A small HTTP service around [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
an open-weight, fast text-to-speech model. Includes support for inline eSpeak
phoneme codes (`[[...]]`) for precise pronunciation control.

The model is loaded once at startup and kept resident for the life of the
process — every request after the first is fast.

## Run it

Locally:

```bash
uv venv --python 3.11 && uv sync   # or: bash setup.sh
uv run server.py                    # PORT=8001 by default
```

With Docker:

```bash
docker build -t kokoro-tts .
docker run -p 8001:8001 -v ~/.cache/huggingface:/root/.cache/huggingface kokoro-tts
```

(Mounting the Hugging Face cache lets the container reuse an already-downloaded
model instead of fetching it again.)

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /synthesize`

Request body:

```json
{
  "text": "Hello world",
  "voice": "af_heart",
  "speed": 1.0,
  "out": "/path/to/out.wav"
}
```

- `text` — required.
- `voice` — a Kokoro voice id (default `af_heart`). May contain inline eSpeak
  phoneme spans, e.g. `Check out [[p'Eks@lz]] today` for precise pronunciation
  of a name/word Kokoro's own G2P gets wrong.
- `speed` — playback speed multiplier (default `1.0`).
- `out` — **optional**. Only meaningful if this service shares a filesystem
  or volume with the caller (e.g. both running in the same docker-compose
  stack). See response shapes below.

**If `out` is given** (co-located mode): the service writes the WAV file to
that path and responds with:

```json
{"ok": true, "sample_rate": 24000, "duration_s": 1.23}
```

**If `out` is omitted** (standalone mode — the normal case for an external
caller): the response body *is* the WAV audio (`Content-Type: audio/wav`),
with metadata in headers:

```
X-Sample-Rate: 24000
X-Duration-Seconds: 1.230
```

**On error**: an HTTP error status (422 for a malformed request, 500 for a
synthesis failure) with a JSON `{"detail": "..."}` body.

## Model license

Kokoro-82M weights are distributed under the Apache 2.0 license by the
[hexgrad](https://huggingface.co/hexgrad) team — see the model card for
details.
