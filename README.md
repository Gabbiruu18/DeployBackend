# AGILA Face API — Cloud Run Starter

This is a **copy‑paste** starter you can deploy to **Cloud Run**.  
It matches your storage layout:

- Embeddings CSV: `faces/features_all.csv`
- Photos: `faces/{role}/{uid}/{name}/face_k.jpg`

## Files
- `app.py` — Flask API (`/register-face`, `/recognize-face`, `/reload-index`, `/health`)
- `requirements.txt` — Python deps (Torch CPU, DeepFace, Ultralytics, OpenCV headless)
- `Dockerfile` — container build
- `deploy.sh` — one‑command build + deploy (Linux/macOS)

## You need
1. **gcloud** SDK installed and logged in:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. Your **Firebase bucket name** (default used here: `agila-c10a4.appspot.com`).
3. The YOLO face weights file in this folder: `yolov8n-face-lindevs.pt`

## Quick deploy (recommended)
```bash
# in this folder
chmod +x deploy.sh
./deploy.sh YOUR_PROJECT_ID
```
- Region defaults to `asia-southeast1`. Change with `./deploy.sh YOUR_PROJECT_ID face-api asia-southeast1`.
- After deploy, the script prints the **service URL**.

## Manual deploy (if you prefer commands)
```bash
# 1) Enable services
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com firestore.googleapis.com storage.googleapis.com

# 2) Create a Docker repo (one time)
gcloud artifacts repositories create face-repo --repository-format=docker --location=asia-southeast1 --description="Face API images"

# 3) Build the image
gcloud builds submit --tag asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/face-repo/face-api

# 4) Deploy to Cloud Run
gcloud run deploy face-api \
  --image asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/face-repo/face-api \
  --region asia-southeast1 \
  --platform managed \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --concurrency 1 \
  --set-env-vars BUCKET_NAME=agila-c10a4.appspot.com,SIM_THRESHOLD=0.65
```

## Give permissions to the service account
```bash
SA_EMAIL=$(gcloud run services describe face-api --region asia-southeast1 --format='value(spec.template.spec.serviceAccountName)')
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID --member="serviceAccount:${SA_EMAIL}" --role="roles/datastore.user"
```

## Test the API
```bash
BASE_URL=$(gcloud run services describe face-api --region asia-southeast1 --format='value(status.url)')

# health
curl -s "$BASE_URL/health"

# reload (forces re-read of faces/features_all.csv)
curl -s -X POST "$BASE_URL/reload-index"

# register (uid plus multiple images)
curl -s -X POST "$BASE_URL/register-face" \
  -F uid=TEST_UID \
  -F image=@/path/to/selfie1.jpg \
  -F image=@/path/to/selfie2.jpg

# recognize (single photo)
curl -s -X POST "$BASE_URL/recognize-face" \
  -F image=@/path/to/photo.jpg
```

## Notes
- **First run may be slow** because DeepFace downloads SFace weights.
- Keep `yolov8n-face-lindevs.pt` next to `app.py` (or set env `YOLO_WEIGHTS=/app/your_path.pt`).
- You can tweak threshold via env: `SIM_THRESHOLD=0.65`.
- If you run multiple Cloud Run instances, call `/reload-index` after a new enrollment so all instances refresh the CSV index.
