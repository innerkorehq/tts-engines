#!/usr/bin/env python
"""
Kokoro TTS synthesis service.

Loads the Kokoro `KPipeline` ONCE at startup and serves synthesis requests
over a small HTTP API (FastAPI) for the lifetime of the process — instead of
paying a full model load on every single call.

Endpoints:
    GET  /health       -> {"status": "ok"} once the model has finished loading.
    POST /synthesize    Request body:
                           {"text": "...", "voice": "af_heart", "speed": 1.0,
                            "out": "/path/to/out.wav" (optional)}
                         If "out" is given, the server writes the WAV file to
                         that path (useful when the caller shares a
                         filesystem/volume with this service, e.g. via
                         docker-compose) and responds with JSON:
                           {"ok": true, "sample_rate": 24000, "duration_s": 1.23}
                         If "out" is omitted, the response body IS the WAV
                         audio (Content-Type: audio/wav), with sample_rate/
                         duration_s in the X-Sample-Rate/X-Duration-Seconds
                         response headers — for any caller that doesn't share
                         a filesystem with this service.
                         On error: an HTTP error status with a JSON
                         {"detail": "..."} body (FastAPI's standard shape).

Run: uv run server.py   (or: uvicorn server:app --host 0.0.0.0 --port 8001)
Env vars: PORT (default 8001), HF_HOME, HF_TOKEN
"""
import io
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kokoro")

_TARGET_SR = 24_000
_REPO = "hexgrad/Kokoro-82M"
_DEFAULT_PORT = 8001


# ── eSpeak phoneme code support via Kokoro/misaki ───────────────────────────
#
# Pronunciation entries may use eSpeak-ng phoneme codes in [[...]] syntax,
# e.g. [[p'Eks@lz]] for "Pexels".  Kokoro's misaki G2P does not interpret
# these natively, so we:
#   1. Split the utterance on [[...]] boundaries.
#   2. Synthesise plain-text spans via the normal Kokoro pipeline.
#   3. Convert [[...]] spans to IPA using the eSpeak→IPA table, then pass
#      the IPA string directly to KPipeline.generate_from_tokens() — which
#      synthesises using the same Kokoro voice model, no extra process.
#   4. Concatenate all 24 kHz PCM segments.

_PHONEME_PATTERN = re.compile(r"(\[\[.*?\]\])", re.DOTALL)

# eSpeak notation → IPA mapping. Order matters for the alternation regex —
# multi-char sequences (diphthongs, affricates) must be tried before
# single-char ones so they're never split apart.
_ESPEAK_MAP: dict = {
    # Diphthongs
    "eI": "eɪ", "aI": "aɪ", "OI": "ɔɪ", "@U": "oʊ", "aU": "aʊ",
    "I@": "ɪə", "e@": "ɛə", "U@": "ʊə",
    # Affricates
    "tS": "tʃ", "dZ": "dʒ",
    # Multi-char consonants
    "T": "θ", "D": "ð", "S": "ʃ", "Z": "ʒ", "N": "ŋ",
    # Vowels (uppercase before lowercase to avoid partial match issues)
    "I": "ɪ", "E": "ɛ", "V": "ʌ", "A": "ɑ", "O": "ɒ", "U": "ʊ",
    "@": "ə", "3": "ɜ", "&": "æ",
    # Base consonants
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "f": "f", "v": "v", "h": "h", "m": "m", "n": "n",
    "l": "l", "r": "ɹ", "w": "w", "j": "j",
    # Base vowels
    "i": "i", "u": "u", "e": "e", "o": "o", "a": "a",
    # Stress markers
    "'": "ˈ", ",": "ˌ",
    # Other
    "?": "ʔ", "x": "x",
}
_ESPEAK_RE = re.compile("|".join(re.escape(k) for k in _ESPEAK_MAP))


def _has_phoneme_codes(text: str) -> bool:
    return "[[" in text and "]]" in text


def _split_phoneme_segments(text: str):
    """
    Split *text* into alternating (kind, value) pairs.

    Example:
        "Check out [[p'Eks@lz]] today"
        → [("text", "Check out "), ("phoneme", "[[p'Eks@lz]]"), ("text", " today")]
    """
    segments = []
    for part in _PHONEME_PATTERN.split(text):
        if not part:
            continue
        kind = "phoneme" if (part.startswith("[[") and part.endswith("]]")) else "text"
        segments.append((kind, part))
    return segments


def _espeak_to_ipa(code: str) -> str:
    """Convert the inner content of [[...]] (eSpeak notation) to IPA."""
    return _ESPEAK_RE.sub(lambda m: _ESPEAK_MAP.get(m.group(), m.group()), code)


def _repo_looks_cached(repo_id: str) -> bool:
    """
    Cheap heuristic: does this repo have at least one snapshot directory in
    the local HF cache? Checked via plain filesystem access — NOT via
    huggingface_hub itself, since importing it (or anything that imports it,
    e.g. kokoro below) before HF_HUB_OFFLINE is set makes the env var a
    no-op (huggingface_hub reads it into a module-level constant at import
    time, not dynamically per call).
    """
    cache_dir = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _load_pipeline():
    # kokoro's KPipeline uses huggingface_hub.hf_hub_download() unconditionally
    # for every file it needs (config/weights/voice packs) — no local-cache
    # check of its own, so without HF_HUB_OFFLINE it hits huggingface.co's API
    # on every single startup even when everything is already cached, and a
    # transient HF Hub connectivity blip crashes startup outright. Force
    # offline mode once the repo is confirmed cached locally.
    if _repo_looks_cached(_REPO):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        logger.info("Not fully cached locally yet — downloading %s…", _REPO)

    from kokoro import KPipeline
    logger.info("Loading Kokoro KPipeline (one-time, first call may trigger model download)…")
    pipeline = KPipeline(lang_code="a")
    logger.info("Kokoro KPipeline ready.")
    return pipeline


def _synthesise_plain(pipeline, text: str, voice: str, speed: float):
    import numpy as np

    audio_chunks = []
    for result in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(result.audio)

    if not audio_chunks:
        raise RuntimeError("Kokoro returned no audio chunks for the provided text.")

    return np.concatenate(audio_chunks)


def _synthesise_mixed(pipeline, text: str, voice: str, speed: float):
    """
    Segment synthesis for text that contains [[eSpeak phoneme codes]].

    Plain-text spans go through the standard Kokoro pipeline. Phoneme spans
    are converted to IPA and passed directly to
    KPipeline.generate_from_tokens() — same Kokoro voice model, no
    subprocess, no sample-rate mismatch.
    """
    import numpy as np

    pcm_parts = []

    for kind, value in _split_phoneme_segments(text):
        chunks = []

        if kind == "phoneme":
            inner = value[2:-2].strip()
            ipa = _espeak_to_ipa(inner)
            if not ipa:
                continue
            logger.debug("eSpeak [[%s]] -> IPA '%s'", inner, ipa)
            for result in pipeline.generate_from_tokens(ipa, voice=voice, speed=speed):
                if result.audio is not None:
                    chunks.append(result.audio)
        else:
            stripped = value.strip()
            if not stripped:
                continue
            for result in pipeline(stripped, voice=voice, speed=speed):
                if result.audio is not None:
                    chunks.append(result.audio)

        if chunks:
            pcm_parts.append(np.concatenate(chunks))

    if not pcm_parts:
        raise RuntimeError(f"No audio produced for text: {text!r}")

    return np.concatenate(pcm_parts)


def _synthesise(pipeline, text: str, voice: str, speed: float):
    """Returns a (sample_rate, mono float32 PCM ndarray) tuple."""
    if _has_phoneme_codes(text):
        audio = _synthesise_mixed(pipeline, text, voice, speed)
    else:
        audio = _synthesise_plain(pipeline, text, voice, speed)
    return _TARGET_SR, audio


# ── FastAPI app ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = _load_pipeline()
    yield


app = FastAPI(title="kokoro-tts", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    out: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok" if "pipeline" in _state else "loading"}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    import soundfile as sf

    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model still loading — retry shortly.")

    try:
        sample_rate, audio = _synthesise(pipeline, req.text, req.voice, req.speed)
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    duration_s = len(audio) / sample_rate

    if req.out:
        # Co-located mode: caller shares a filesystem/volume with this
        # service (e.g. docker-compose) and just wants the file written —
        # same contract the original stdin/stdout daemon offered.
        Path(req.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(req.out, audio, sample_rate)
        return {"ok": True, "sample_rate": sample_rate, "duration_s": duration_s}

    # Standalone mode: no shared filesystem assumed — return the audio
    # itself in the response body.
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Duration-Seconds": f"{duration_s:.3f}",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", _DEFAULT_PORT)))
