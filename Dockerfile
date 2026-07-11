# StyleForge — Track 2 submission image (Participant Guide contract).
# Build (Apple Silicon -> required linux/amd64 manifest):
#   docker buildx build --platform linux/amd64 \
#     --build-arg FIREWORKS_API_KEY=$FIREWORKS_API_KEY -t <registry>/styleforge:latest --push .
# Harness runs it with /input/tasks.json mounted; we write /output/results.json and exit 0.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    # Prebuilt CPU wheel — avoids compiling llama.cpp under qemu in buildx
    && pip install --no-cache-dir llama-cpp-python \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# The DPO-tuned Gemma 3 4B (Q4_K_M GGUF) — round-2 model (won the eval vs v1).
COPY data/models/styleforge-v2-q4.gguf /app/models/styleforge-gemma-q4.gguf

# Pre-fetch the Whisper model at build time — no download inside the 10-min eval window.
ENV HF_HOME=/app/.cache
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"

# Track 2 injects NO env vars; credentials ship in the image (accepted risk on a
# capped hackathon key — rotate after the event).
ARG FIREWORKS_API_KEY=""
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

ENV STYLEFORGE_DATA=/data
RUN mkdir -p /data /input /output

ENTRYPOINT ["styleforge"]
CMD ["harness"]
