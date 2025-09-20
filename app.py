import os, io, re, csv, tempfile, json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud.exceptions import NotFound

# ------------------------
# Config
# ------------------------
BUCKET_NAME = os.getenv("BUCKET_NAME", "agila-c10a4.firebasestorage.app")
EMBEDDINGS_CSV_PATH = "faces/features_all.csv"
MODEL_NAME = "SFace"
YOLO_WEIGHTS = os.getenv("yolov8n-face-lindevs.pt")
THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.65"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.5"))  # detection confidence
MAX_ENROLL_IMAGES = int(os.getenv("MAX_ENROLL_IMAGES", "10"))

ROLES = ["student", "teacher", "program_head", "academic_head"]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ------------------------
# Firebase init (ADC on Cloud Run)
# ------------------------
if not firebase_admin._apps:
    # Prefer default App Engine / Cloud Run credentials (no file needed)
    firebase_admin.initialize_app(options={"storageBucket": BUCKET_NAME})
db = firestore.client()
bucket = storage.bucket()

# ------------------------
# Models (load once per instance)
# ------------------------
device = "cpu"
torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "2"))))
model = YOLO(YOLO_WEIGHTS)
# set confidence
try:
    model.model.conf = YOLO_CONF  # ultralytics v8 attr may vary; fallback to passing conf in call
except Exception:
    pass

# ------------------------
# In-memory index
# ------------------------
known_matrix: Optional[np.ndarray] = None  # shape (N, D), L2-normalized
labels: List[Tuple[str, str, str]] = []    # (uid, name, role)
feature_dim: Optional[int] = None

# ------------------------
# Helpers
# ------------------------
def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)

def safe_name(name: str) -> str:
    # keep letters, numbers, space, dash, underscore → replace spaces with underscore, drop others
    s = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip()
    return re.sub(r"\s+", "_", s) or "user"

def find_user_by_uid(uid: str) -> Optional[Tuple[str, str]]:
    """Return (name, role) from Firestore or None if not found."""
    for role in ROLES:
        ref = db.collection("users").document(role).collection("accounts").document(uid)
        snap = ref.get()
        if snap.exists:
            d = snap.to_dict() or {}
            first = str(d.get("firstName", "")).strip()
            last = str(d.get("lastName", "")).strip()
            name = (first + " " + last).strip() or d.get("name") or uid
            return name, role
    return None

def detect_faces_bboxes_bgr(img_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    with torch.no_grad():
        res = model(img_bgr, verbose=False, conf=YOLO_CONF)[0]
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    # ensure ints and sane boxes
    H, W = img_bgr.shape[:2]
    out = []
    for (x1,y1,x2,y2) in boxes:
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(W,x2), min(H,y2)
        if x2 > x1 and y2 > y1:
            out.append((x1,y1,x2,y2))
    return out

def crop_with_margin(img: np.ndarray, box: Tuple[int,int,int,int], margin: float=0.2) -> np.ndarray:
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    w, h = x2 - x1, y2 - y1
    x1m = max(0, int(x1 - margin * w))
    y1m = max(0, int(y1 - margin * h))
    x2m = min(W, int(x2 + margin * w))
    y2m = min(H, int(y2 + margin * h))
    return img[y1m:y2m, x1m:x2m]

def face_to_embedding(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    if face_bgr is None or face_bgr.size == 0:
        return None
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (160, 160))
    # DeepFace.represent returns list of dicts
    rep = DeepFace.represent(face_rgb, model_name=MODEL_NAME,
                            detector_backend="skip", enforce_detection=False)
    if not rep:
        return None
    emb = np.asarray(rep[0]["embedding"], dtype=np.float32)
    return emb

def read_embeddings_csv() -> pd.DataFrame:
    blob = bucket.blob(EMBEDDINGS_CSV_PATH)
    try:
        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data))
        return df
    except NotFound:
        return pd.DataFrame(columns=["uid", "name", "role"] + [f"feature_{i}" for i in range(512)])
    except Exception as e:
        print(f"[WARN] Could not read CSV: {e}")
        # return empty with default 512 cols
        return pd.DataFrame(columns=["uid", "name", "role"] + [f"feature_{i}" for i in range(512)])

def write_embeddings_csv(df: pd.DataFrame) -> None:
    # Ensure feature_* columns contiguous and ordered
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    feature_cols_sorted = sorted(feature_cols, key=lambda c: int(c.split("_")[1]))
    cols = ["uid", "name", "role"] + feature_cols_sorted
    csv_bytes = df[cols].to_csv(index=False).encode("utf-8")
    bucket.blob(EMBEDDINGS_CSV_PATH).upload_from_string(csv_bytes, content_type="text/csv")

def rebuild_index_from_csv() -> Tuple[int, Optional[int]]:
    global known_matrix, labels, feature_dim
    df = read_embeddings_csv()
    if df.empty:
        known_matrix = None
        labels = []
        feature_dim = None
        return 0, None

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        known_matrix = None
        labels = []
        feature_dim = None
        return 0, None

    feature_cols_sorted = sorted(feature_cols, key=lambda c: int(c.split("_")[1]))
    mat = df[feature_cols_sorted].to_numpy(dtype=np.float32)
    mat = l2_normalize(mat, axis=1)

    names = list(df["name"].astype(str))
    uids = list(df["uid"].astype(str))
    roles = list(df["role"].astype(str))
    labs = list(zip(uids, names, roles))

    known_matrix = mat
    labels = labs
    feature_dim = mat.shape[1]
    return mat.shape[0], feature_dim

def upsert_embedding_row(uid: str, name: str, role: str, vec: np.ndarray) -> None:
    df = read_embeddings_csv()

    # detect dim from vector
    d = int(vec.shape[0])
    feature_cols = [f"feature_{i}" for i in range(d)]
    # ensure df has these columns
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # remove extra feature_* columns that exceed d
    extra_cols = [c for c in df.columns if c.startswith("feature_") and int(c.split("_")[1]) >= d]
    if extra_cols:
        # keep them if needed; otherwise we can drop. We'll keep to avoid losing data from others with same dim.
        pass

    # Build new row
    row = {"uid": uid, "name": name, "role": role}
    for i in range(d):
        row[f"feature_{i}"] = float(vec[i])

    if (df["uid"] == uid).any():
        df.loc[df["uid"] == uid, list(row.keys())] = list(row.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    write_embeddings_csv(df)

def add_to_memory(uid: str, name: str, role: str, vec: np.ndarray) -> None:
    global known_matrix, labels, feature_dim
    vec = vec.astype(np.float32)
    vec = l2_normalize(vec)
    if known_matrix is None:
        known_matrix = vec.reshape(1, -1)
        feature_dim = known_matrix.shape[1]
        labels[:] = [(uid, name, role)]
    else:
        # pad dims if mismatch (shouldn't happen if consistent)
        if vec.shape[0] != known_matrix.shape[1]:
            # skip inconsistent vector
            print("[WARN] vector dim mismatch; skipping in-memory add")
            return
        known_matrix = np.vstack([known_matrix, vec[None, :]])
        labels.append((uid, name, role))

def best_match(emb: np.ndarray) -> Tuple[float, int]:
    """Return (best_similarity, index)."""
    global known_matrix
    if known_matrix is None or known_matrix.size == 0:
        return -1.0, -1
    emb = emb.astype(np.float32)
    emb = l2_normalize(emb)
    sims = known_matrix @ emb  # cosine similarity
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx

# ------------------------
# Startup: build index
# ------------------------
count, dim = rebuild_index_from_csv()
print(f"[INIT] Loaded index: {count} vectors, dim={dim}")

# ------------------------
# Endpoints
# ------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "vectors": 0 if known_matrix is None else int(known_matrix.shape[0])}), 200

@app.route("/reload-index", methods=["POST", "GET"])
def reload_index():
    n, d = rebuild_index_from_csv()
    return jsonify({"status": "reloaded", "count": n, "dim": d}), 200

@app.route("/register-face", methods=["POST"])
def register_face():
    # Inputs
    uid = request.form.get("uid", "").strip()
    images = request.files.getlist("image")

    if not uid or not images:
        return jsonify({"status": "error", "message": "uid and at least one image are required"}), 200

    # Lookup canonical name/role from Firestore
    user = find_user_by_uid(uid)
    if not user:
        return jsonify({"status": "error", "message": "uid not found in Firestore"}), 200
    name, role = user
    safe = safe_name(name)

    # Limit images
    images = images[:MAX_ENROLL_IMAGES]

    # Process images → embeddings
    embs = []
    accepted = 0
    for idx, file in enumerate(images, start=1):
        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        boxes = detect_faces_bboxes_bgr(img)
        if not boxes:
            continue
        # choose largest face
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        box = boxes[int(np.argmax(areas))]
        face = crop_with_margin(img, box, margin=0.2)
        emb = face_to_embedding(face)
        if emb is None:
            continue
        embs.append(emb)
        accepted += 1

        # Save crop to Storage
        # Re-encode to JPEG
        ok, jpg = cv2.imencode(".jpg", face)
        if ok:
            path = f"faces/{role}/{uid}/{safe}/face_{idx}.jpg"
            bucket.blob(path).upload_from_string(jpg.tobytes(), content_type="image/jpeg")

    if not embs:
        return jsonify({"status": "enrollment_failed", "reason": "no_valid_faces"}), 200

    # Average + normalize
    embs = np.stack(embs, axis=0)
    avg = embs.mean(axis=0).astype(np.float32)
    avg = l2_normalize(avg)

    # Upsert into CSV and memory
    upsert_embedding_row(uid, name, role, avg)
    add_to_memory(uid, name, role, avg)

    # Mark user as registered
    db.collection("users").document(role).collection("accounts").document(uid).set(
        {"faceRegistered": True}, merge=True
    )

    return jsonify({"status": "enrolled", "uid": uid, "name": name, "role": role, "imagesAccepted": accepted}), 200

@app.route("/recognize-face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image provided"}), 200

    # Decode image
    data = np.frombuffer(request.files["image"].read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"status": "error", "message": "Invalid image"}), 200

    # If no index yet
    if known_matrix is None or known_matrix.size == 0:
        return jsonify({"status": "not_ready"}), 200

    boxes = detect_faces_bboxes_bgr(frame)
    if not boxes:
        return jsonify({"status": "no_face_found"}), 200

    best = {"sim": -1.0, "box": None, "idx": -1}
    for (x1, y1, x2, y2) in boxes:
        face = crop_with_margin(frame, (x1, y1, x2, y2), margin=0.2)
        emb = face_to_embedding(face)
        if emb is None:
            continue
        sim, idx = best_match(emb)
        if sim > best["sim"]:
            best = {"sim": sim, "box": [int(x1), int(y1), int(x2), int(y2)], "idx": idx}

    if best["idx"] >= 0 and best["sim"] >= THRESHOLD:
        uid, name, role = labels[best["idx"]]
        return jsonify({
            "status": "recognized",
            "uid": uid,
            "name": name,
            "role": role,
            "similarity": round(float(best["sim"]), 3),
            "bounding_box": best["box"]
        }), 200

    return jsonify({"status": "not_recognized"}), 200

# Cloud Run entry
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
