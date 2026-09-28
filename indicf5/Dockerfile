FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv venv --python 3.11 && uv sync --frozen

COPY server.py ./

ENV PORT=8011
EXPOSE 8011

# ai4bharat/IndicF5 is gated on Hugging Face — pass HF_TOKEN at runtime:
#   docker run -e HF_TOKEN=... -p 8011:8011 indicf5
CMD [".venv/bin/python", "server.py"]
