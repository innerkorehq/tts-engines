# indicf5

An HTTP service around [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5),
an F5-TTS-based zero-shot voice-cloning model for Indian languages.

> **Gated model**: `ai4bharat/IndicF5` requires accepting its license on
> Hugging Face and an `HF_TOKEN` with access — see the
> [model page](https://huggingface.co/ai4bharat/IndicF5).
>
> **Known environment caveat**: pulls in `torchaudio`, which (in recent
> versions) depends on `torchcodec` for audio loading, which only supports
> ffmpeg versions 4-8 — see the sibling `f5tts` engine's README for details
> if `/synthesize` fails with a `libtorchcodec` error.

The model is loaded once at startup and kept resident for the life of the
process. Includes two runtime patches applied to IndicF5's own
`trust_remote_code` model file and its vendored `f5_tts` dependency:
- Adds an Apple Silicon (MPS) code path — IndicF5's upstream code only
  checks for CUDA and otherwise falls back to (slow) CPU.
- Fixes an audio-duration miscalculation for Roman-script input relative to
  Devanagari reference text (byte-count vs character-count mismatch).

## Run it

```bash
export HF_TOKEN=hf_...   # ai4bharat/IndicF5 is gated
uv venv --python 3.13 && uv sync   # or: bash setup.sh
uv run server.py                    # PORT=8011 by default
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
  "text": "नमस्ते दुनिया",
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

See the [model card](https://huggingface.co/ai4bharat/IndicF5) for
licensing terms.
