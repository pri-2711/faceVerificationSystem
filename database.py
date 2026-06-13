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
from pymongo.errors import ConfigurationError, InvalidURI, OperationFailure, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
import numpy as np
import config


# ── Persistent connection ─────────────────────────────────────


def _is_atlas_uri(uri: str) -> bool:
    return uri.startswith("mongodb+srv://") or "mongodb.net" in uri


def _connection_hint(exc: Exception) -> str:
    uri = config.MONGO_URI or ""
    atlas = _is_atlas_uri(uri)
    message = str(exc).lower()

    if isinstance(exc, InvalidURI):
        return (
            "The MongoDB URI is malformed. If the username or password contains "
            "special characters such as @, :, /, ?, #, or %, they must be URL-encoded."
        )

    if isinstance(exc, ConfigurationError):
        if "dnspython" in message:
            return (
                "Atlas SRV URIs require the dnspython package. Add dnspython to requirements.txt and redeploy."
            )
        return (
            "MongoDB configuration is invalid. Verify the URI format, cluster settings, and the target database name."
        )

    if isinstance(exc, ServerSelectionTimeoutError):
        if atlas:
            return (
                "MongoDB Atlas could not be reached. Check that the cluster is online, the database user credentials are correct, "
                "the password is URL-encoded if it contains special characters, your IP address is added to Atlas Network Access, "
                "and the user has access to the target database."
            )
        return (
            "The local MongoDB server could not be reached. Confirm the service is running and that the URI points to the correct host and port."
        )

    if isinstance(exc, OperationFailure):
        if "auth" in message or "authentication" in message:
            return (
                "MongoDB authentication failed. Verify the Atlas database user, password, and any URL encoding needed for special characters."
            )
        return (
            "MongoDB rejected the operation. Verify the database user permissions and that the target database exists."
        )

    if atlas:
        return (
            "Atlas connection failed. Recheck the URI, secrets, network access allowlist, and database credentials."
        )

    return (
        "MongoDB connection failed. Verify the URI, database name, and that the MongoDB server is running."
    )


@st.cache_resource(show_spinner=False)
def _get_client():
    """Create a reusable MongoDB client once per Streamlit session."""
    if not config.MONGO_URI:
        raise RuntimeError(
            "MONGO_URI is not set. Add it to Streamlit Secrets or environment variables."
        )

    return MongoClient(
        config.MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
        retryWrites=True,
        retryReads=True,
        appname="face-verification-system",
        server_api=ServerApi("1"),
    )


def _get_db():
    """
    Resolve the MongoDB database and verify the server is reachable.

    The client itself is cached, but the ping runs on each rerun so a transient
    startup failure does not get stuck in the UI.
    """
    try:
        if not config.DB_NAME:
            raise RuntimeError(
                "DB_NAME is not set. Add it to Streamlit Secrets or environment variables."
            )

        client = _get_client()
        client.admin.command("ping")
        print("[DB] MongoDB connected.")
        return client[config.DB_NAME]
    except Exception as exc:
        source = "Atlas/secret or env" if _is_atlas_uri(config.MONGO_URI) else "local/secret or env"
        st.error(
            "**MongoDB connection failed.**\n\n"
            f"URI source: `{source}`\n\n"
            f"Database: `{config.DB_NAME or '(missing)'}`\n\n"
            f"Details: `{exc}`\n\n"
            f"What to check: {_connection_hint(exc)}"
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