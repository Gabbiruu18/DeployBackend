    #!/usr/bin/env bash
    # Deploy Face API to Cloud Run (CPU)
    # Usage examples:
    #   bash deploy.sh
    #   bash deploy.sh --auth private --max-instances 1
    #   bash deploy.sh --registry ar --repo face-api-repo --service face-api-dev

    set -euo pipefail

    # -------- Defaults (override via flags) --------
    PROJECT_ID="lithe-site-472709-p5"
    REGION="asia-southeast1"
    SERVICE="face-api"
    CPU="2"
    MEMORY="2Gi"
    PORT="8080"
    CONCURRENCY="1"          # safer for CPU-bound workloads
    MIN_INSTANCES="0"        # keep costs low on free tier
    MAX_INSTANCES="2"
    AUTH="public"            # public | private
    REGISTRY="gcr"           # gcr | ar (Artifact Registry)
    REPO="face-api"          # AR repo name if --registry ar
    BUCKET_NAME="agila-c10a4.appspot.com"
    SIM_THRESHOLD="0.60"
    YOLO_WEIGHTS="models/yolov8n-face-lindevs.pt"
    SERVICE_ACCOUNT=""       # optional: pass an explicit SA email
    GRANT_IAM="false"        # grant Firestore/Storage to SA
    LABELS="service=agila,env=dev"

    # -------- Parse flags --------
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --project) PROJECT_ID="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --service) SERVICE="$2"; shift 2 ;;
        --cpu) CPU="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --min-instances) MIN_INSTANCES="$2"; shift 2 ;;
        --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --auth) AUTH="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --bucket) BUCKET_NAME="$2"; shift 2 ;;
        --threshold) SIM_THRESHOLD="$2"; shift 2 ;;
        --weights) YOLO_WEIGHTS="$2"; shift 2 ;;
        --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
        --grant-iam) GRANT_IAM="true"; shift 1 ;;
        --labels) LABELS="$2"; shift 2 ;;
        -h|--help)
          cat <<EOF
Usage: bash deploy.sh [options]

Options:
  --project <id>            GCP project id (default: ${PROJECT_ID})
  --region <region>         Cloud Run region (default: ${REGION})
  --service <name>          Service name (default: ${SERVICE})
  --cpu <n>                 vCPU (default: ${CPU})
  --memory <size>           memory (default: ${MEMORY})
  --port <n>                container port (default: ${PORT})
  --concurrency <n>         requests per instance (default: ${CONCURRENCY})
  --min-instances <n>       min instances (default: ${MIN_INSTANCES})
  --max-instances <n>       max instances (default: ${MAX_INSTANCES})
  --auth public|private     public allows unauthenticated (default: ${AUTH})
  --registry gcr|ar         container registry (default: ${REGISTRY})
  --repo <name>             AR repo name if --registry ar (default: ${REPO})
  --bucket <name>           Cloud Storage bucket (default: ${BUCKET_NAME})
  --threshold <float>       similarity threshold (default: ${SIM_THRESHOLD})
  --weights <path>          YOLO weights path in image (default: ${YOLO_WEIGHTS})
  --service-account <email> Service account to run as (optional)
  --grant-iam               Grant Storage + Firestore roles to SA (optional)
  --labels <k=v,...>        Comma-separated labels (default: ${LABELS})
EOF
          exit 0
          ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
      esac
    done

    log()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
    warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
    die()  { echo -e "\033[1;31m[ERR ]\033[0m $*" >&2; exit 1; }

    # -------- Pre-flight checks --------
    command -v gcloud >/dev/null 2>&1 || die "gcloud not found. Use Cloud Shell or install the SDK."
    [[ -f "Dockerfile" ]] || die "Dockerfile not found in current directory."
    [[ -f "app.py" ]] || die "app.py not found in current directory."
    [[ -f "${YOLO_WEIGHTS}" ]] || die "Weights not found at ${YOLO_WEIGHTS}. Place your PT file there."

    # -------- Configure gcloud --------
    log "Setting project: ${PROJECT_ID}"
    gcloud config set project "${PROJECT_ID}" >/dev/null

    # -------- Choose image URL --------
    TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
    if [[ "${REGISTRY}" == "ar" ]]; then
      # Artifact Registry
      IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG}"
      if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
        log "Creating Artifact Registry repo: ${REPO} in ${REGION}"
        gcloud artifacts repositories create "${REPO}" --repository-format=docker --location="${REGION}" --description="Images for ${SERVICE}"
      fi
    else
      # Container Registry (legacy)
      IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}:${TAG}"
    fi

    # -------- Build image --------
    log "Building image: ${IMAGE}"
    gcloud builds submit --tag "${IMAGE}"

    # -------- Determine service account --------
    if [[ -z "${SERVICE_ACCOUNT}" ]]; then
      PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
      SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
      log "Using default service account: ${SERVICE_ACCOUNT}"
    else
      log "Using provided service account: ${SERVICE_ACCOUNT}"
    fi

    if [[ "${GRANT_IAM}" == "true" ]]; then
      log "Granting Firestore + Storage roles to ${SERVICE_ACCOUNT}"
      gcloud projects add-iam-policy-binding "${PROJECT_ID}"         --member="serviceAccount:${SERVICE_ACCOUNT}" --role="roles/datastore.user" >/dev/null
      gcloud projects add-iam-policy-binding "${PROJECT_ID}"         --member="serviceAccount:${SERVICE_ACCOUNT}" --role="roles/storage.objectAdmin" >/dev/null
    fi

    # -------- Deploy to Cloud Run --------
    AUTH_FLAG="--allow-unauthenticated"
    [[ "${AUTH}" == "private" ]] && AUTH_FLAG="--no-allow-unauthenticated"

    log "Deploying service: ${SERVICE} to ${REGION} (${AUTH})"
    gcloud run deploy "${SERVICE}"       --image "${IMAGE}"       --region "${REGION}"       --platform managed       --cpu "${CPU}"       --memory "${MEMORY}"       --port "${PORT}"       --concurrency "${CONCURRENCY}"       --min-instances "${MIN_INSTANCES}"       --max-instances "${MAX_INSTANCES}"       --labels "${LABELS}"       --service-account "${SERVICE_ACCOUNT}"       ${AUTH_FLAG}       --set-env-vars "BUCKET_NAME=${BUCKET_NAME},SIM_THRESHOLD=${SIM_THRESHOLD},YOLO_WEIGHTS=${YOLO_WEIGHTS}"

    URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"
    log "Deployed URL: ${URL}"

    # -------- Show test commands --------
    echo
    echo "=== Quick test ==="
    if [[ "${AUTH}" == "private" ]]; then
      echo "TOKEN=\$(gcloud auth print-identity-token)"
      echo "curl -H "Authorization: Bearer \$TOKEN" -s ${URL}/ | jq"
      echo "# Register:"
      echo "curl -H "Authorization: Bearer \$TOKEN" -X POST ${URL}/register-face \
  -F "uid=abc123" -F "name=Jane Doe" -F "role=student" \
  -F "image=@/path/1.jpg" -F "image=@/path/2.jpg""
      echo "# Recognize:"
      echo "curl -H "Authorization: Bearer \$TOKEN" -X POST ${URL}/recognize-face -F "image=@/path/test.jpg""
    else
      echo "curl -s ${URL}/ | jq"
      echo "# Register:"
      echo "curl -X POST ${URL}/register-face \
  -F "uid=abc123" -F "name=Jane Doe" -F "role=student" \
  -F "image=@/path/1.jpg" -F "image=@/path/2.jpg""
      echo "# Recognize:"
      echo "curl -X POST ${URL}/recognize-face -F "image=@/path/test.jpg""
    fi
    echo
    log "Done."
