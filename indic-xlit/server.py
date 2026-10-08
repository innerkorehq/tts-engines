#!/usr/bin/env python
"""
OPTIONAL HTTP wrapper around engine.py, for standalone use only. vidgen does
not use this — it calls engine.py in-process (no HTTP, no ports).

    GET  /health   -> {"status": "ok" | "loading"}
    POST <engine.ROUTE>  JSON request as documented in engine.py
        -> JSON result, or the WAV bytes (audio/wav, with X-Sample-Rate /
           X-Duration-Seconds headers) when the request has no "out".

Run: uv run --extra http server.py      (PORT env var; see engine.DEFAULT_PORT)
"""
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Response

import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield


app = FastAPI(title=engine.NAME, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if engine.is_ready() else "loading"}


@app.post(engine.ROUTE)
async def call(payload: dict):
    # Runs on the event-loop thread (the same thread load() ran on) — required
    # by engines whose GPU stream is thread-local (MLX); calls are serialized.
    try:
        result = engine.run(payload)
    except engine.EngineError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    wav = result.pop("wav_bytes", None)
    if wav is not None:
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(result.get("sample_rate", "")),
                "X-Duration-Seconds": f"{result.get('duration_s', 0.0):.3f}",
            },
        )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", engine.DEFAULT_PORT)))
