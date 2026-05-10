# =============================================================
#  register.py  —  All registration logic
#
#  Three modes:
#    1. Single person  — from a folder of images
#    2. Single person  — live from webcam
#    3. Bulk           — entire processed_data/ dataset at once
# =============================================================

import os
import cv2
import numpy as np
from embedding  import get_single_embedding
from similarity import normalize
from database   import save_user, user_exists
import config


# ── Helpers ───────────────────────────────────────────────────

def _average_embeddings(embeddings):
    """Average a list of embeddings and L2-normalize the result."""
    avg = np.mean(embeddings, axis=0)
    return normalize(avg)


def _embeddings_from_folder(folder_path):
    """
    Read all images from a folder, extract one embedding per image.
    Returns list of valid embeddings and count of skipped files.
    """
    supported = (".jpg", ".jpeg", ".png")
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(supported)
    ]

    embeddings = []
    skipped    = 0

    for fname in files:
        img = cv2.imread(os.path.join(folder_path, fname))
        if img is None:
            skipped += 1
            continue

        emb = get_single_embedding(img)
        if emb is None:
            skipped += 1
        else:
            embeddings.append(emb)

    return embeddings, skipped


# ── Mode 1: Single person from folder ─────────────────────────

def register_from_folder(set_id, name, folder_path):
    """
    Register one person using images from a given folder.
    All embeddings are averaged into a single representative vector.

    Args:
        set_id      (str) : unique ID (e.g. "S001")
        name        (str) : display name
        folder_path (str) : path containing face images

    Returns:
        True on success, False on failure
    """
    print(f"\n[Register] Processing: {name} ({set_id})")

    if not os.path.isdir(folder_path):
        print(f"[Register] Folder not found: {folder_path}")
        return False

    embeddings, skipped = _embeddings_from_folder(folder_path)

    if not embeddings:
        print(f"[Register] No valid faces found in folder. Skipped {skipped} file(s).")
        return False

    avg_emb = _average_embeddings(embeddings)
    save_user(set_id, name, avg_emb)

    print(f"[Register] Done — {len(embeddings)} image(s) used, {skipped} skipped.")
    return True


# ── Mode 2: Single person from webcam ─────────────────────────

def register_from_webcam(set_id, name, num_captures=10):
    """
    Register one person by capturing live frames from webcam.
    Press SPACE to capture, Q to quit early.

    Args:
        set_id       (str) : unique ID
        name         (str) : display name
        num_captures (int) : number of frames to capture
    """
    print(f"\n[Register] Webcam registration for: {name} ({set_id})")
    print(f"[Register] Press SPACE to capture ({num_captures} needed), Q to quit.\n")

    cap        = cv2.VideoCapture(0)
    embeddings = []
    count      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Register] Webcam error.")
            break

        overlay = frame.copy()
        cv2.putText(overlay, f"Captures: {count}/{num_captures}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        cv2.putText(overlay, "SPACE = Capture    Q = Quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.imshow(f"Register — {name}", overlay)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            emb = get_single_embedding(frame)
            if emb is None:
                print("[Register] No face detected — try again.")
            else:
                embeddings.append(emb)
                count += 1
                print(f"[Register] Captured {count}/{num_captures}")
                if count >= num_captures:
                    print("[Register] All captures done.")
                    break

        elif key == ord("q"):
            print("[Register] Stopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings:
        print("[Register] No valid captures — registration cancelled.")
        return False

    avg_emb = _average_embeddings(embeddings)
    save_user(set_id, name, avg_emb)
    print(f"[Register] Registered {name} with {len(embeddings)} capture(s).")
    return True


# ── Mode 3: Bulk registration from processed_data/ ────────────

def register_bulk(dataset_path=None, skip_existing=True):
    """
    Register all persons found in the dataset folder.

    Folder structure expected:
        processed_data/
            person_1/
                ID_1.jpg
                Selfie_1.jpg
                Selfie_2.jpg
            person_2/
                ...

    The folder name becomes both set_id and name
    (e.g. folder "person_1" → set_id="person_1", name="person_1").

    Args:
        dataset_path   (str)  : path to processed_data folder
                                defaults to config.DATASET_PATH
        skip_existing  (bool) : if True, skip persons already in DB

    Returns:
        dict with keys: total, registered, skipped, failed
    """
    path = dataset_path or config.DATASET_PATH

    if not os.path.isdir(path):
        print(f"[Bulk] Dataset folder not found: {path}")
        return {"total": 0, "registered": 0, "skipped": 0, "failed": 0}

    # Each subfolder = one person
    person_folders = sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ])

    total      = len(person_folders)
    registered = 0
    skipped    = 0
    failed     = 0

    print(f"\n[Bulk] Found {total} person folder(s) in {path}\n")

    for folder_name in person_folders:
        set_id      = folder_name          # e.g. "person_1"
        name        = folder_name
        folder_path = os.path.join(path, folder_name)

        # Skip if already in DB
        if skip_existing and user_exists(set_id):
            print(f"[Bulk] Skipping (already registered): {set_id}")
            skipped += 1
            continue

        success = register_from_folder(set_id, name, folder_path)

        if success:
            registered += 1
        else:
            failed += 1

    print(f"\n[Bulk] Complete — {registered} registered, {skipped} skipped, {failed} failed.")
    return {
        "total":      total,
        "registered": registered,
        "skipped":    skipped,
        "failed":     failed
    }


# ── Quick CLI run ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Face Registration ===")
    print("1. Register one person (from folder)")
    print("2. Register one person (from webcam)")
    print("3. Bulk register entire dataset")
    choice = input("\nChoose (1/2/3): ").strip()

    if choice == "1":
        sid    = input("Enter ID (e.g. S001): ").strip()
        name   = input("Enter name: ").strip()
        folder = input("Enter image folder path: ").strip()
        register_from_folder(sid, name, folder)

    elif choice == "2":
        sid  = input("Enter ID: ").strip()
        name = input("Enter name: ").strip()
        register_from_webcam(sid, name)

    elif choice == "3":
        register_bulk()

    else:
        print("Invalid choice.")