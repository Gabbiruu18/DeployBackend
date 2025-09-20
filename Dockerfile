FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 ffmpeg libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from deploybackend/ into /app
COPY . .
COPY app/models/yolov8n-face-lindevs.pt /app/models/

ENV PORT=8080

# If your file is app.py and has: app = Flask(__name__)
CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --threads 8
# If it's main.py, change to:  CMD exec gunicorn main:app --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --threads 8
