FROM python:3.11-slim

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu

# Copy app + weights
COPY . .
# Ensure yolov8n-face-lindevs.pt is present in the repo root (or change YOLO_WEIGHTS env)

ENV PORT=8080
ENV BUCKET_NAME=agila-c10a4.appspot.com
# Optional: ENV SIM_THRESHOLD=0.65

# Start with gunicorn (good for prod)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 120 app:app
