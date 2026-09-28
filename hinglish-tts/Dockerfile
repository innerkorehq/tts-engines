FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY inference.py setup.sh ./
COPY scoring ./scoring
RUN bash setup.sh

COPY server.py ./

ENV PORT=8005
EXPOSE 8005

# ai4bharat/IndicF5 is gated on Hugging Face — pass HF_TOKEN at runtime:
#   docker run -e HF_TOKEN=... -p 8005:8005 hinglish-tts
CMD [".venv/bin/python", "server.py"]
