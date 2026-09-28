# NOTE: this engine depends on mlx-audio (Apple's MLX framework), which only
# ships wheels for Apple Silicon (macOS arm64) — `uv sync` below will likely
# fail to even install mlx inside this Linux container. This Dockerfile is
# kept for structural consistency across engines, not because it currently
# works. Run this engine natively on an Apple Silicon Mac instead:
#   uv sync && uv run server.py
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv venv --python 3.11 && uv sync --frozen

COPY server.py ./

ENV PORT=8002
EXPOSE 8002

CMD [".venv/bin/python", "server.py"]
