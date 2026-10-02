# chatterbox-multilingual-hi

An HTTP service around [gagan1985/chatterbox-multilingual-hi-mlx-fp16](https://huggingface.co/gagan1985/chatterbox-multilingual-hi-mlx-fp16),
an MLX (Apple Silicon) port of Resemble AI's Chatterbox-Multilingual model,
fine-tuned/packaged for Hindi zero-shot voice cloning.

> **Apple Silicon only.** This engine depends on [mlx-audio](https://github.com/Blaizzy/mlx-audio),
> which requires Apple's MLX framework — macOS on Apple Silicon (M-series).
> It will not install or run on Linux/CUDA/Windows.

The model is loaded once at startup and kept resident for the life of the
process. Repeat requests for the same voice additionally skip the expensive
reference-audio-to-speaker-embedding pass via an on-disk cache (see below).

## Run it

```bash
uv venv --python 3.13 && uv sync   # or: bash setup.sh
uv run server.py                    # PORT=8002 by default
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
  "text": "नमस्ते, यह एक परीक्षण है",
  "ref_audio": "/path/to/reference.wav",
  "voice_id": "vp_some_stable_id",
  "out": "/path/to/out.wav"
}
```

- `text` — required.
- `ref_audio` — path to a reference audio clip for zero-shot voice cloning
  (optional; omit to use the model's own default voice, if it has one baked
  in via a bundled `conds.safetensors`).
- `voice_id` — **optional but recommended**: a stable identifier for this
  reference voice (e.g. a voice-profile id from your own system). When set,
  the expensive `ref_audio` → speaker-conditioning pass (S3 tokenizer twice,
  `s3gen.embed_ref`, the voice encoder — real forward passes) is computed
  once and cached on disk under `.cache/speaker_conds/<voice_id>.safetensors`,
  keyed by a cheap fingerprint of `ref_audio` (size + mtime) so a later
  request with the same `voice_id` reuses it instantly — a big speedup when
  synthesizing many clips for the same voice. Omitting `voice_id` falls back
  to recomputing the speaker conditioning on every single request.
- `out` — **optional**. Only meaningful if this service shares a filesystem
  or volume with the caller. See response shapes below.

**If `out` is given** (co-located mode): the service writes the WAV file to
that path and responds with:

```json
{"ok": true, "sample_rate": 24000, "duration_s": 1.23}
```

**If `out` is omitted** (standalone mode): the response body *is* the WAV
audio (`Content-Type: audio/wav`), with metadata in headers:

```
X-Sample-Rate: 24000
X-Duration-Seconds: 1.230
```

**On error**: an HTTP error status (422 for a malformed request, 500 for a
synthesis failure) with a JSON `{"detail": "..."}` body.

## Model license

See the [model card](https://huggingface.co/gagan1985/chatterbox-multilingual-hi-mlx-fp16)
for licensing terms of the underlying Chatterbox-Multilingual weights.
