FROM python:3.10-slim

# System libs for OpenCV/FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Flask app (the file that contains "app = Flask(__name__)")
# If your file is named something else, keep names consistent below.
COPY . .

# Cloud Run passes PORT
ENV PORT=8080

# Start with gunicorn for production
# Replace main:app if your module is not main.py
CMD exec gunicorn main:app --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --threads 8
