
# app.py — Cloud Run (lazy init, crash-proof)
import os, io, time, tempfile, threading
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from ultralytics import YOLO
from deepface import DeepFace
import firebase_admin
from firebase_admin import firestore, storage

PORT = int(os.environ.get("PORT", "8080"))
BUCKET_NAME = os.environ.get("BUCKET_NAME", "agila-c10a4.appspot.com")
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "models/yolov8n-face-lindevs.pt")
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.60"))

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
detector = None               # lazy
bucket = None                 # lazy
db = None                     # lazy

face_features_known_list: List[np.ndarray] = []
face_name_known_list: List[str] = []
uid_cache = {}
_db_lock = threading.Lock()

def log(*a): print("[app]", *a, flush=True)

def init_firebase():
    global db, bucket
    if not firebase_admin._apps:
        opts = {"storageBucket": BUCKET_NAME} if BUCKET_NAME else None
        firebase_admin.initialize_app(options=opts)
    if db is None: db = firestore.client()
    if bucket is None: bucket = storage.bucket()

def get_detector():
    global detector
    if detector is None:
        log("Loading YOLO weights:", YOLO_WEIGHTS, "on", device)
        detector = YOLO(YOLO_WEIGHTS)
        detector.to(device)
    return detector

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / den)

def get_name_from_uid(uid: str) -> Tuple[str, str, str]:
    if uid in uid_cache: return uid_cache[uid]
    try:
        for role in ["student","teacher","program_head","academic_head"]:
            doc = db.collection("users").document(role).collection("accounts").document(uid).get()
            if doc.exists:
                d = doc.to_dict() or {}
                first = d.get("firstName") or d.get("first_name") or ""
                last = d.get("lastName") or d.get("last_name") or ""
                uid_cache[uid] = (last, first, role); return uid_cache[uid]
    except Exception as e:
        log("get_name_from_uid error:", e)
    return ("Unknown", "", "unknown")

def load_face_database():
    global face_features_known_list, face_name_known_list
    try:
        init_firebase()
        with _db_lock:
            face_features_known_list.clear(); face_name_known_list.clear()
            blob = bucket.blob("faces/features_all.csv")
            if not blob.exists():
                log("features_all.csv missing — OK (empty DB)")
                return
            data = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(data))
            for _, row in df.iterrows():
                try:
                    uid = str(row.get("uid","")).strip()
                    last, first, role = get_name_from_uid(uid)
                    vec = np.array([float(row[f"feature_{i}"]) for i in range(128)], dtype=np.float32)
                    face_name_known_list.append(f"{last}|{first}|{role}|{uid}")
                    face_features_known_list.append(vec)
                except Exception as ex:
                    log("row skip:", ex)
    except Exception as e:
        # Never crash startup even if bucket/creds are wrong
        log("load_face_database error:", e)

@app.get("/")
def root():
    # Try a best-effort lazy warm without crashing
    try:
        _ = get_detector()
    except Exception as e:
        log("Detector not ready:", e)
    if not face_name_known_list:
        try: load_face_database()
        except Exception as e: log("Warm DB error:", e)
    return jsonify({"status":"ok","faces_cached":len(face_name_known_list)})

def _safe_resize(bgr, size=(160,160)):
    try: return cv2.resize(bgr, size)
    except Exception: return None

def _embedding_from_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rep = DeepFace.represent(img_path=face_rgb, model_name="SFace",
                             enforce_detection=False, detector_backend="skip")
    return np.array(rep[0]["embedding"], dtype=np.float32)

@app.post("/register-face")
def register_face():
    try:
        init_firebase()
        det = get_detector()
        uid = (request.form.get("uid") or "").strip()
        name = (request.form.get("name") or "").strip()
        role = (request.form.get("role") or "").strip()
        images = request.files.getlist("image")
        if not uid or not role or not images:
            return jsonify({"success":False,"message":"uid, role, images required"}), 400

        embs = []
        ts = int(time.time())
        with tempfile.TemporaryDirectory() as tmp:
            for i, file in enumerate(images):
                path = os.path.join(tmp, f"in_{i}.jpg"); file.save(path)
                img = cv2.imread(path); 
                if img is None: continue
                results = det(img, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy().astype(int) if results.boxes is not None else []
                if len(boxes)==0: continue
                x1,y1,x2,y2 = boxes[0].tolist()
                face = img[y1:y2, x1:x2]
                face = _safe_resize(face)
                if face is None or face.size==0: continue
                try:
                    embs.append(_embedding_from_face(face))
                    bucket.blob(f"faces/{uid}/images/{ts}_{i}.jpg").upload_from_filename(path, content_type="image/jpeg")
                except Exception as ex:
                    log("embedding/snapshot error:", ex)
                    continue

        if not embs:
            return jsonify({"success":False,"message":"No valid face detected"}), 400
        avg = np.mean(np.stack(embs, axis=0), axis=0)

        blob = bucket.blob("faces/features_all.csv")
        if blob.exists():
            df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
        else:
            df = pd.DataFrame(columns=["uid","name","role"] + [f"feature_{i}" for i in range(128)])
        row = {"uid": uid, "name": name or uid, "role": role}
        for i in range(128): row[f"feature_{i}"] = float(avg[i])
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        out = io.BytesIO(); df.to_csv(out, index=False); out.seek(0)
        blob.upload_from_file(out, content_type="text/csv")

        # Firestore flag (non-fatal on error)
        try:
            firestore.client().collection("users").document(role).collection("accounts").document(uid).set(
                {"faceRegistered": True, "faceRegisteredAt": firestore.SERVER_TIMESTAMP}, merge=True)
        except Exception as ex:
            log("firestore flag error:", ex)

        # refresh cache
        load_face_database()
        return jsonify({"success":True,"message":f"{len(embs)} image(s) processed"}), 200

    except Exception as e:
        log("register-face error:", e)
        return jsonify({"success":False,"message":str(e)}), 500

@app.post("/recognize-face")
def recognize_face():
    try:
        det = get_detector()
        if not face_features_known_list:
            load_face_database()
        if "image" not in request.files:
            return jsonify({"error":"No image provided"}), 400
        npimg = np.frombuffer(request.files["image"].read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error":"Invalid image"}), 400
        results = det(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int) if results.boxes is not None else []
        best = {"sim":-1.0,"box":None,"who":None}
        for box in boxes:
            x1,y1,x2,y2 = box.tolist()
            crop = frame[y1:y2, x1:x2]; crop = _safe_resize(crop)
            if crop is None or crop.size==0: continue
            try:
                emb = _embedding_from_face(crop)
            except Exception as ex:
                log("embedding error:", ex); continue
            sims = [cosine_similarity(emb, k) for k in face_features_known_list]
            if sims:
                s = float(np.max(sims)); idx = int(np.argmax(sims))
                if s > best["sim"]:
                    best = {"sim":s, "box":(int(x1),int(y1),int(x2),int(y2)), "who":face_name_known_list[idx]}
        if best["sim"] >= SIM_THRESHOLD and best["who"]:
            last, first, role, uid = best["who"].split("|")
            return jsonify({"status":"recognized","uid":uid,"name":f"{first} {last}".strip(),"role":role,
                            "similarity":round(best["sim"],3),"bounding_box":list(best["box"])}), 200
        return jsonify({"status":"not recognized"}), 404
    except Exception as e:
        log("recognize-face error:", e)
        return jsonify({"success":False,"message":str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
