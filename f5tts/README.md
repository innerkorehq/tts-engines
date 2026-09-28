# f5tts

An HTTP service around [F5-TTS](https://github.com/SWivid/F5-TTS)
(`F5TTS_v1_Base`), an open zero-shot voice-cloning TTS model.

The model + vocoder are loaded once at startup and kept resident for the
life of the process. Reference-audio preprocessing (silence trimming/
re-export) is cached per `(ref_audio, ref_text)` pair, so repeat requests
for the same voice skip that work.

> **Known environment caveat**: this engine's audio-loading dependency
> (`torchcodec`, pulled in transitively by `torchaudio`) only supports
> ffmpeg versions 4-8. If your system has ffmpeg 9+ installed (e.g. a recent
> Homebrew install), `/synthesize` will fail with a `libtorchcodec` load
> error. Install a compatible ffmpeg version (4-8) alongside, or wait for an
> updated `torchcodec` release with ffmpeg 9 support.

## Run it

```bash
uv venv --python 3.11 && uv sync   # or: bash setup.sh
uv run server.py                    # PORT=8003 by default
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
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "The transcript of the reference clip",
  "remove_silence": true,
  "nfe_step": 32,
  "cfg_strength": 2.0,
  "out": "/path/to/out.wav"
}
```

- `text`, `ref_audio`, `ref_text` — required. `ref_text` is the transcript
  of `ref_audio` (F5-TTS needs both the audio and its text to clone a voice).
- `remove_silence` — post-process the output to trim silence (default `true`).
- `nfe_step` / `cfg_strength` — quality/speed trade-off knobs; defaults match
  upstream F5-TTS defaults.
- `out` — **optional**. Only meaningful if this service shares a filesystem
  or volume with the caller.

**If `out` is given**: the service writes the WAV file to that path and
responds with `{"ok": true}`.

**If `out` is omitted**: the response body *is* the WAV audio
(`Content-Type: audio/wav`).

**On error**: an HTTP error status with a JSON `{"detail": "..."}` body.

## Model license

See the [F5-TTS repository](https://github.com/SWivid/F5-TTS) and the
[model card](https://huggingface.co/SWivid/F5-TTS) for licensing terms.
