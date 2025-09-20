#!/usr/bin/env bash
# Simple build & deploy script for Cloud Run (Asia Southeast 1)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./deploy.sh <GCP_PROJECT_ID> [SERVICE_NAME=face-api] [REGION=asia-southeast1]"
  exit 1
fi

PROJECT_ID="$1"
SERVICE_NAME="${2:-face-api}"
REGION="${3:-asia-southeast1}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/face-repo/${SERVICE_NAME}"

# Enable required services
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com firestore.googleapis.com storage.googleapis.com --project "${PROJECT_ID}"

# Create Artifact Registry (if not exists)
gcloud artifacts repositories create face-repo --repository-format=docker --location="${REGION}" --description="Face API images" --project "${PROJECT_ID}" || true

# Build image
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}"


# add YOLO_WEIGHTS to --set-env-vars if you didn't use /app/models/...
--set-env-vars BUCKET_NAME=agila-c10a4.appspot.com,SIM_THRESHOLD=0.65,YOLO_WEIGHTS=/app/models/yolov8n-face-lindevs.pt


# Deploy to Cloud Run
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --concurrency 1 \
  --set-env-vars BUCKET_NAME=agila-c10a4.appspot.com,SIM_THRESHOLD=0.65 \
  --project "${PROJECT_ID}"

# Show URL
echo "Deployed. Service URL:"
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)' --project "${PROJECT_ID}"

# Grant IAM to service account (Storage + Firestore)
SA_EMAIL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(spec.template.spec.serviceAccountName)' --project "${PROJECT_ID}")
echo "Granting IAM roles to ${SA_EMAIL}"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${SA_EMAIL}" --role="roles/datastore.user"

echo "Done."
