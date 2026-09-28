# supertonic

An HTTP service around [Supertonic TTS 3](https://github.com/supertone-inc/supertonic),
an ONNX-based, CPU-only TTS model — 44.1kHz output, 31 languages, 10
preset voices (`M1`-`M5`, `F1`-`F5`).

The model is loaded once at startup (auto-downloading weights on first run
if needed) and kept resident for the life of the process. Runs entirely on
CPU via ONNX Runtime — no GPU/MPS/MLX dependency, so this is the one engine
in this repo that runs anywhere (Linux, macOS, Windows).

## Run it

```bash
uv venv --python 3.11 && uv sync   # or: bash setup.sh
uv run server.py                    # PORT=8009 by default
```

With Docker:

```bash
docker build -t supertonic-tts .
docker run -p 8009:8009 supertonic-tts
```

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
  "voice": "M1",
  "lang": "en",
  "speed": 1.05,
  "steps": 8,
  "out": "/path/to/out.wav"
}
```

- `text` — required.
- `voice` — one of the 10 preset voices, `M1`-`M5` / `F1`-`F5` (default `M1`).
- `lang` — language code (default `na`, meaning auto/language-agnostic).
- `speed` — playback speed multiplier (default `1.05`).
- `steps` — diffusion steps; higher is slower but can improve quality
  (default `8`).
- `out` — **optional**. Only meaningful if this service shares a filesystem
  or volume with the caller.

**If `out` is given**: the service writes the WAV file to that path and
responds with:

```json
{"ok": true, "sample_rate": 44100, "duration_s": 1.23}
```

**If `out` is omitted**: the response body *is* the WAV audio
(`Content-Type: audio/wav`), with metadata in headers
(`X-Sample-Rate`, `X-Duration-Seconds`).

**On error**: an HTTP error status (422 for empty text, 500 for a
synthesis failure, 503 while the model is still loading) with a JSON
`{"detail": "..."}` body.

## License

See the [Supertonic repository](https://github.com/supertone-inc/supertonic)
for licensing terms.
