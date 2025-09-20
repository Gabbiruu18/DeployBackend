
# app.py — Cloud Run–ready
import os, io, csv, tempfile, time, threading
from typing import List, Tuple
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
from deepface import DeepFace
import firebase_admin
from firebase_admin import firestore, storage

# ------------------------
# Config via environment
# ------------------------
PORT = int(os.environ.get("PORT", "8080"))
BUCKET_NAME = os.environ.get("BUCKET_NAME", "agila-c10a4.firebasestorage.app")
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "models/yolov8n-face-lindevs.pt")
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.60"))  # tweak in Cloud Run env vars

# ------------------------
# Firebase (use ADC on Cloud Run)
# ------------------------
if not firebase_admin._apps:
    firebase_admin.initialize_app(options={"storageBucket": BUCKET_NAME})

db = firestore.client()
bucket = storage.bucket()

# ------------------------
# Flask app
# ------------------------
app = Flask(__name__)
CORS(app)

# ------------------------
# YOLO model (CPU on Cloud Run)
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = YOLO(YOLO_WEIGHTS)
detector.to(device)

# ------------------------
# Face DB cache
# ------------------------
face_features_known_list: List[np.ndarray] = []
face_name_known_list: List[str] = []
uid_cache = {}
_db_lock = threading.Lock()

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def get_name_from_uid(uid: str) -> Tuple[str, str, str]:
    if uid in uid_cache:
        return uid_cache[uid]
    # Search across roles
    for role in ["student", "teacher", "program_head", "academic_head"]:
        doc = db.collection("users").document(role).collection("accounts").document(uid).get()
        if doc.exists:
            d = doc.to_dict() or {}
            first = d.get("firstName", "") or d.get("first_name", "")
            last = d.get("lastName", "") or d.get("last_name", "")
            uid_cache[uid] = (last, first, role)
            return uid_cache[uid]
    return ("Unknown", "", "unknown")

def load_face_database():
    """Load features_all.csv from Cloud Storage into memory."""
    global face_features_known_list, face_name_known_list
    with _db_lock:
        face_features_known_list.clear()
        face_name_known_list.clear()

        blob = bucket.blob("faces/features_all.csv")
        if not blob.exists():
            return

        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data))

        for _, row in df.iterrows():
            try:
                uid = str(row.get("uid", "")).strip()
                last, first, role = get_name_from_uid(uid)
                # 128-dim SFace embedding (DeepFace)
                features = np.array([float(row[f"feature_{i}"]) for i in range(128)], dtype=np.float32)
                face_name_known_list.append(f"{last}|{first}|{role}|{uid}")
                face_features_known_list.append(features)
            except Exception as e:
                print(f"[load_face_database] skip row: {e}")

# Warm-load on instance start
load_face_database()

def _safe_resize(bgr, size=(160, 160)):
    try:
        return cv2.resize(bgr, size)
    except Exception:
        return None

def _represent_sface(face_bgr):
    # DeepFace expects RGB by default; we set detector_backend='skip'
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rep = DeepFace.represent(
        img_path = face_rgb,
        model_name = "SFace",
        enforce_detection = False,
        detector_backend = "skip"
    )
    return np.array(rep[0]["embedding"], dtype=np.float32)

@app.get("/")
def root():
    return jsonify({"status": "ok", "faces_cached": len(face_name_known_list)})

@app.post("/register-face")
def register_face():
    try:
        uid = request.form.get("uid", "").strip()
        name = request.form.get("name", "").strip()
        role = request.form.get("role", "").strip()
        images = request.files.getlist("image")

        if not uid or not role or not images:
            return jsonify({"success": False, "message": "uid, role and at least one image are required"}), 400

        embeddings = []
        ts = int(time.time())

        with tempfile.TemporaryDirectory() as tmp:
            for i, file in enumerate(images):
                path = os.path.join(tmp, f"in_{i}.jpg")
                file.save(path)
                img = cv2.imread(path)
                if img is None:
                    continue

                # detect face
                results = detector(img, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy().astype(int) if results.boxes is not None else []
                if len(boxes) == 0:
                    continue

                x1, y1, x2, y2 = boxes[0].tolist()
                face = img[y1:y2, x1:x2]
                face = _safe_resize(face)
                if face is None or face.size == 0:
                    continue

                # compute embedding
                try:
                    emb = _represent_sface(face)
                    embeddings.append(emb)
                    # store snapshot
                    snap_blob = bucket.blob(f"faces/{uid}/images/{ts}_{i}.jpg")
                    snap_blob.upload_from_filename(path, content_type="image/jpeg")
                except Exception as e:
                    print("[register-face] embedding failed:", e)
                    continue

        if not embeddings:
            return jsonify({"success": False, "message": "No valid face detected."}), 400

        avg = np.mean(np.stack(embeddings, axis=0), axis=0)

        # Update features_all.csv in memory and in GCS
        # Download existing (if any)
        blob = bucket.blob("faces/features_all.csv")
        if blob.exists():
            buf = io.BytesIO(blob.download_as_bytes())
            df = pd.read_csv(buf)
        else:
            df = pd.DataFrame(columns=["uid", "name", "role"] + [f"feature_{i}" for i in range(128)])

        row = {"uid": uid, "name": name or f"{uid}", "role": role}
        for i in range(128):
            row[f"feature_{i}"] = float(avg[i])
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        out = io.BytesIO()
        df.to_csv(out, index=False)
        out.seek(0)
        blob.upload_from_file(out, content_type="text/csv")

        # Mark user in Firestore
        db.collection("users").document(role).collection("accounts").document(uid).set(
            {"faceRegistered": True, "faceRegisteredAt": firestore.SERVER_TIMESTAMP},
            merge=True,
        )

        # refresh cache for this instance
        load_face_database()

        return jsonify({"success": True, "message": f"{len(embeddings)} image(s) processed"}), 200

    except Exception as e:
        print("[register-face] error:", e)
        return jsonify({"success": False, "message": str(e)}), 500

@app.post("/recognize-face")
def recognize_face():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_bytes = request.files["image"].read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        results = detector(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int) if results.boxes is not None else []

        if not face_features_known_list:
            # lazy load in case this instance missed warm load
            load_face_database()

        best = {"sim": -1.0, "box": None, "who": None}
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            crop = frame[y1:y2, x1:x2]
            crop = _safe_resize(crop)
            if crop is None or crop.size == 0:
                continue

            try:
                emb = _represent_sface(crop)
            except Exception as e:
                print("[recognize-face] embedding failed:", e)
                continue

            sims = [cosine_similarity(emb, known) for known in face_features_known_list]
            if sims:
                s = max(sims)
                idx = int(np.argmax(np.array(sims)))
                if s > best["sim"]:
                    best["sim"] = s
                    best["box"] = (int(x1), int(y1), int(x2), int(y2))
                    best["who"] = face_name_known_list[idx]

        if best["sim"] >= SIM_THRESHOLD and best["who"]:
            last, first, role, uid = best["who"].split("|")
            return jsonify({
                "status": "recognized",
                "uid": uid,
                "name": f"{first} {last}".strip(),
                "role": role,
                "similarity": round(float(best["sim"]), 3),
                "bounding_box": list(best["box"]),
            }), 200

        return jsonify({"status": "not recognized"}), 404

    except Exception as e:
        print("[recognize-face] error:", e)
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    # Local dev only; Cloud Run uses gunicorn CMD
    app.run(host="0.0.0.0", port=PORT)
