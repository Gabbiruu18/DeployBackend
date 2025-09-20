
# Face API on Cloud Run (CPU)

## 1) Prepare model weights
Put `yolov8n-face-lindevs.pt` inside `models/` beside `app.py` (already created for you).

## 2) Build & deploy
Replace PROJECT_ID and region if needed.
```bash
gcloud config set project lithe-site-472709-p5
gcloud builds submit --tag gcr.io/lithe-site-472709-p5/face-api

gcloud run deploy face-api       --image gcr.io/lithe-site-472709-p5/face-api       --region asia-southeast1       --allow-unauthenticated       --cpu 2 --memory 2Gi       --port 8080       --set-env-vars BUCKET_NAME=agila-c10a4.appspot.com,SIM_THRESHOLD=0.60,YOLO_WEIGHTS=models/yolov8n-face-lindevs.pt
```

> Tip: If you prefer authenticated-only access, remove `--allow-unauthenticated` and call with an identity token:
> `curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" ...`

## 3) Test
```bash
# Health
curl -s https://YOUR_URL/ | jq

# Register (multipart/form-data)
curl -X POST https://YOUR_URL/register-face       -F "uid=abc123" -F "name=Jane Doe" -F "role=student"       -F "image=@/path/to/1.jpg" -F "image=@/path/to/2.jpg"

# Recognize
curl -X POST https://YOUR_URL/recognize-face       -F "image=@/path/to/test.jpg"
```

## Notes
- Cloud Run standard instances have no GPUs; this image uses CPU-only Torch.
- The app reads/writes `faces/features_all.csv` in **Cloud Storage** (bucket given by `BUCKET_NAME`).
- Firebase Admin uses **Application Default Credentials** on Cloud Run (no JSON path in code).
- You can tune performance by increasing CPU/memory and the similarity threshold via env vars.
