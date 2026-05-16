FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version first
RUN pip install torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi==0.111.0 uvicorn==0.30.1 \
    python-multipart==0.0.9 Pillow==10.3.0 \
    google-generativeai==0.7.2 edge-tts==6.1.12 \
    huggingface-hub==0.23.4 nest-asyncio==1.6.0

COPY app.py .

EXPOSE ${PORT:-8000}
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
