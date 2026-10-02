FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv venv --python 3.13 && uv sync --frozen

COPY server.py ./

ENV PORT=8004
EXPOSE 8004

CMD [".venv/bin/python", "server.py"]
