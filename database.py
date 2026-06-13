# =============================================================
#  database.py  —  All MongoDB read / write operations
#
#  Collections:
#    users    → registered users + their embeddings
#    logs     → every verification event
#    unknowns → snapshots of unrecognised faces
# =============================================================

import streamlit as st
from pymongo import MongoClient
from datetime import datetime, timedelta
import numpy as np
import config


# ── Persistent connection ─────────────────────────────────────

def _get_client():
    """Create the MongoDB client once per Streamlit session."""
    return MongoClient(
        config.MONGO_URI,
        serverSelectionTimeoutMS = 5000,
        connectTimeoutMS         = 5000,
    )


def _get_db():
    """
    Resolve the MongoDB database and verify the server is reachable.

    The client itself is cached, but the ping runs on each rerun so a
    transient startup failure does not get stuck in the UI.
    """
    try:
        client = _get_client()
        client.admin.command("ping")   # confirms connection is live
        print("[DB] MongoDB connected.")
        return client[config.DB_NAME]
    except Exception as exc:
        st.error(
            "**MongoDB connection failed.**\n\n"
            f"URI: `{config.MONGO_URI}`\n\n"
            f"Details: `{exc}`\n\n"
            "If MongoDB was just started, refresh the app. Otherwise check\n"
            "the URI in `config.py` and confirm the MongoDB service is up."
        )
        st.stop()


def _cols():
    """Return (users_col, logs_col, unknowns_col) from the cached DB."""
    db = _get_db()
    return db["users"], db["logs"], db["unknowns"]


# ── Users ──────────────────────────────────────────────────────

def save_user(set_id, name, embedding):
    users_col, _, _ = _cols()
    users_col.update_one(
        {"set_id": set_id},
        {"$set": {
            "name":          name,
            "embedding":     embedding.tolist(),
            "registered_at": datetime.now()
        }},
        upsert=True
    )


def load_all_users():
    users_col, _, _ = _cols()
    return [
        {
            "set_id":    doc["set_id"],
            "name":      doc["name"],
            "embedding": np.array(doc["embedding"])
        }
        for doc in users_col.find()
    ]


def user_exists(set_id):
    users_col, _, _ = _cols()
    return users_col.find_one({"set_id": set_id}) is not None


def delete_user(set_id):
    users_col, _, _ = _cols()
    return users_col.delete_one({"set_id": set_id}).deleted_count > 0


def count_users():
    users_col, _, _ = _cols()
    return users_col.count_documents({})


# ── Logs ───────────────────────────────────────────────────────

def save_log(identity, status, score, image_b64=None):
    _, logs_col, _ = _cols()
    logs_col.insert_one({
        "identity":  identity,
        "status":    status,
        "score":     round(float(score), 4),
        "timestamp": datetime.now(),
        "image_b64": image_b64
    })


def get_recent_logs(limit=50):
    _, logs_col, _ = _cols()
    logs = list(logs_col.find().sort("timestamp", -1).limit(limit))
    for log in logs:
        log.pop("_id", None)
    return logs


# ── Unknowns ───────────────────────────────────────────────────

def save_unknown(image_b64, source="webcam"):
    _, _, unknowns_col = _cols()
    unknowns_col.insert_one({
        "image_b64": image_b64,
        "source":    source,
        "timestamp": datetime.now()
    })


def count_unknowns_in_window(minutes):
    _, _, unknowns_col = _cols()
    cutoff = datetime.now() - timedelta(minutes=minutes)
    return unknowns_col.count_documents({"timestamp": {"$gte": cutoff}})