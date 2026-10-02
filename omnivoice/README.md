# omnivoice

An HTTP service around [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
(the official PyTorch package, not an MLX port) — a massively multilingual
(600+ languages) zero-shot TTS model.

The model is loaded once at startup and kept resident for the life of the
process. On Apple Silicon it runs on MPS by default, with an automatic
one-way fallback to CPU if MPS runs out of memory (sticky for the rest of
the process's life once triggered). Also includes a retry loop for a known
short-text-degeneration issue in this model, detected via a peak/RMS
heuristic on the generated audio.

## Run it

```bash
uv venv --python 3.13 && uv sync   # or: bash setup.sh
uv run server.py                    # PORT=8008 by default
```

Model weights download lazily from `k2-fsa/OmniVoice` on first startup.

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
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "Optional transcript of the reference clip",
  "out": "/path/to/out.wav"
}
```

- `text` — required.
- `ref_audio` — optional, for zero-shot voice cloning.
- `ref_text` — optional; if `ref_audio` is given without it, the model
  auto-transcribes the reference clip via Whisper.
- `out` — **optional**. Only meaningful if this service shares a filesystem
  or volume with the caller.

**If `out` is given**: the service writes the WAV file to that path and
responds with:

```json
{"ok": true, "sample_rate": 24000, "duration_s": 1.23}
```

**If `out` is omitted**: the response body *is* the WAV audio
(`Content-Type: audio/wav`), with metadata in headers
(`X-Sample-Rate`, `X-Duration-Seconds`).

**On error**: an HTTP error status (422 for empty text, 500 for a
synthesis failure, 503 while the model is still loading) with a JSON
`{"detail": "..."}` body.

## Model license

See the [model card](https://huggingface.co/k2-fsa/OmniVoice) for licensing
terms.
