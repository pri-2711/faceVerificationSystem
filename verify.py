# =============================================================
#  verify.py  —  All verification logic
#
#  Two modes:
#    1. Webcam  — real-time frame-by-frame detection
#    2. Image   — single photo or group photo (uploaded file)
#
#  Both modes:
#    → extract embeddings via embedding.py
#    → match via similarity.py
#    → log to MongoDB via database.py
#    → trigger email alert via alert.py when unknown
# =============================================================

import cv2
import base64
import numpy as np
from datetime import datetime

from embedding  import get_all_embeddings, get_single_embedding
from similarity import find_best_match
from database   import (
    load_all_users, save_log,
    save_unknown, count_unknowns_in_window
)
from alert import alert_unknown_face, alert_multiple_unknowns, alert_duplicate_entry
import config

# In-memory duplicate tracking  { name: datetime of last verified entry }
_recent_entries = {}


# ── Helpers ───────────────────────────────────────────────────

def _is_duplicate(name):
    if name in _recent_entries:
        elapsed = (datetime.now() - _recent_entries[name]).total_seconds()
        return elapsed < config.DUPLICATE_WINDOW_SEC
    return False


def _frame_to_bytes(frame):
    """Encode OpenCV frame to raw JPEG bytes (for email attachment)."""
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _frame_to_b64(frame):
    """Encode OpenCV frame to base64 string (for MongoDB storage)."""
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode("utf-8")


def _draw_box(frame, box, name, score, status):
    """Draw bounding box and name/score label on frame."""
    x1, y1, x2, y2 = [int(c) for c in box]
    color = (0, 200, 0) if status == "Verified" else (0, 0, 220)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{name}  {score:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def _handle_unknown(frame, score, source="webcam"):
    """Log, save snapshot, and send alert for an unknown face."""
    ts          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_bytes = _frame_to_bytes(frame)
    face_b64    = base64.b64encode(image_bytes).decode("utf-8")

    save_log("Unknown", "Unknown", score, face_b64)
    save_unknown(face_b64, source=source)
    alert_unknown_face(timestamp=ts, image_bytes=image_bytes)

    # Escalated alert if too many unknowns in window
    count = count_unknowns_in_window(config.UNKNOWN_ALERT_WINDOW)
    if count >= config.UNKNOWN_ALERT_COUNT:
        alert_multiple_unknowns(count, config.UNKNOWN_ALERT_WINDOW, image_bytes)

    print(f"[Verify] Unknown face. Score: {score:.4f}  Source: {source}")


def _handle_verified(name, set_id, score, frame=None):
    """Log a verified entry; flag and alert if duplicate."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if _is_duplicate(name):
        alert_duplicate_entry(name, timestamp=ts)
        print(f"[Verify] Duplicate entry: {name}")
        return "Duplicate"

    _recent_entries[name] = datetime.now()
    face_b64 = None
    if frame is not None:
        face_b64 = _frame_to_b64(frame)
    save_log(name, "Verified", score, face_b64)
    print(f"[Verify] Verified: {name}  Score: {score:.4f}")
    return "Verified"


# ── Mode 1: Webcam verification ───────────────────────────────

def run_webcam(registered_users=None):
    """
    Real-time webcam verification loop.
    Loads registered users from DB once, then processes each frame.
    Press Q to quit.

    Args:
        registered_users (list | None) : pre-loaded user list
                                         (if None, loads from DB)
    """
    users = registered_users or load_all_users()

    if not users:
        print("[Verify] No registered users in database. Register faces first.")
        return

    print(f"[Verify] Loaded {len(users)} user(s). Starting webcam — press Q to quit.\n")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Verify] Webcam read error.")
            break

        embeddings, boxes = get_all_embeddings(frame)

        for emb, box in zip(embeddings, boxes):
            name, set_id, score = find_best_match(emb, users)

            if name == "Unknown":
                _draw_box(frame, box, "Unknown", score, "Unknown")
                _handle_unknown(frame, score, source="webcam")
            else:
                status = _handle_verified(name, set_id, score)
                label  = name if status == "Verified" else f"Duplicate: {name}"
                _draw_box(frame, box, label, score, "Verified")

        cv2.imshow("Face Verification  —  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Verify] Webcam session ended.")


# ── Mode 2: Image verification (single or group photo) ────────

def verify_image(image_input, registered_users=None, source="upload"):
    """
    Verify one or more faces in a static image.
    Works for both single-person and group photos.

    Args:
        image_input      : numpy BGR array (from cv2.imread or uploaded file)
        registered_users : pre-loaded user list (if None, loads from DB)
        source           : "upload" or "webcam_snapshot"

    Returns:
        list of dicts — one per detected face:
            {name, set_id, score, status, box}
    """
    users = registered_users or load_all_users()

    if not users:
        print("[Verify] No registered users in database.")
        return []

    embeddings, boxes = get_all_embeddings(image_input)

    if not embeddings:
        print("[Verify] No faces detected in the image.")
        return []

    results = []

    for emb, box in zip(embeddings, boxes):
        name, set_id, score = find_best_match(emb, users)

        if name == "Unknown":
            _handle_unknown(image_input, score, source=source)
            results.append({
                "name":   "Unknown",
                "set_id": None,
                "score":  score,
                "status": "Unknown",
                "box":    box
            })
        else:
            status = _handle_verified(name, set_id, score, image_input)
            results.append({
                "name":   name,
                "set_id": set_id,
                "score":  score,
                "status": status,
                "box":    box
            })

    return results


# ── Quick CLI run ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Face Verification ===")
    print("1. Webcam (real-time)")
    print("2. Image file")
    choice = input("\nChoose (1/2): ").strip()

    if choice == "1":
        run_webcam()
    elif choice == "2":
        path = input("Enter image path: ").strip()
        img  = cv2.imread(path)
        if img is None:
            print("Could not read image.")
        else:
            res = verify_image(img)
            for r in res:
                print(r)
    else:
        print("Invalid choice.")