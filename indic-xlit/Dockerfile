FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock setup.sh ./
RUN bash setup.sh

COPY server.py ./

ENV PORT=8006
EXPOSE 8006

CMD [".venv/bin/python", "server.py"]
