# f5tts-hinglish

An HTTP service around [rajputsw/F5-TTS-Hinglish](https://huggingface.co/rajputsw/F5-TTS-Hinglish),
an F5-TTS checkpoint fine-tuned for Hindi-English code-switched ("Hinglish")
narration.

The model is loaded once at startup and kept resident for the life of the
process.

> **Known environment caveat**: shares the same `torchcodec`/ffmpeg-version
> constraint as the sibling `f5tts` engine — see its README for details.

## Run it

```bash
uv venv --python 3.13 && uv sync   # or: bash setup.sh
uv run server.py                    # PORT=8004 by default
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
  "text": "Hello world, aaj ka din bahut accha hai",
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "The transcript of the reference clip",
  "out": "/path/to/out.wav"
}
```

- `text`, `ref_audio`, `ref_text` — required.
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

**On error**: an HTTP error status with a JSON `{"detail": "..."}` body.

## Model license

See the [model card](https://huggingface.co/rajputsw/F5-TTS-Hinglish) for
licensing terms.
