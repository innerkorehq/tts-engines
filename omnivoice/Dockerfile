FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv venv --python 3.11 && uv sync --frozen

COPY server.py ./

ENV PORT=8008
EXPOSE 8008

CMD [".venv/bin/python", "server.py"]
