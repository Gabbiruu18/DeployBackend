
# Dockerfile for Cloud Run
FROM python:3.10-slim

# System deps for OpenCV / Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Performance & reliability
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and models (place your .pt in models/)
COPY agila_backend.py ./agila_backend.py
COPY models ./models

ENV PORT=8080
EXPOSE 8080

# Use gunicorn in Cloud Run
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-b", "0.0.0.0:8080", "--threads", "4", "--timeout", "120", "agila_backend:app"]
