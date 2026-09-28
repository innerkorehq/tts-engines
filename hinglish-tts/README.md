# hinglish-tts

An HTTP service around [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5)
(via the vendored [harrrshall/hinglish-tts](https://github.com/harrrshall/hinglish-tts)
inference code), tuned for Hindi-English code-switched ("Hinglish") speech.

`inference.py` and `scoring/` are vendored **verbatim** from the upstream
project — this repo only wraps them in a warm HTTP service instead of a
cold-start-per-call script.

> **Gated model**: `ai4bharat/IndicF5` requires accepting its license on
> Hugging Face and an `HF_TOKEN` with access — see the
> [model page](https://huggingface.co/ai4bharat/IndicF5).
>
> **Known environment caveat**: pulls in `torchaudio`/`torchcodec`, which
> only supports ffmpeg versions 4-8 — see the sibling `f5tts` engine's
> README for details if `/synthesize` fails with a `libtorchcodec` error.

## Run it

```bash
export HF_TOKEN=hf_...   # ai4bharat/IndicF5 is gated
bash setup.sh
uv run server.py          # PORT=8005 by default
```

(`setup.sh` installs via a curated set of `pip install` commands mirroring
upstream's own install instructions — there's no `pyproject.toml`/lockfile
for this engine, by design, since several of its dependencies need
`--no-deps`/patched installs that a normal resolver can't express.)

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

See the [ai4bharat/IndicF5 model card](https://huggingface.co/ai4bharat/IndicF5)
for licensing terms.
