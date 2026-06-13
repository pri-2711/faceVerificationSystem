"""Application settings sourced from Streamlit secrets or environment variables.

MongoDB settings are production-first:
1. Read MONGO_URI and DB_NAME from Streamlit secrets.
2. Fall back to environment variables.
3. Allow localhost only when USE_LOCAL_MONGO_FALLBACK is enabled locally.
"""

from __future__ import annotations

import os


def _secret_or_env(name: str, default=None):
	try:
		import streamlit as st

		if name in st.secrets:
			return st.secrets[name]
	except Exception:
		pass

	value = os.getenv(name)
	if value is None or value == "":
		return default
	return value


def _env_flag(name: str, default: bool = False) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


_allow_local_mongo = _env_flag("USE_LOCAL_MONGO_FALLBACK", False)


# ----- MongoDB -----------------------------------------------
MONGO_URI = _secret_or_env(
	"MONGO_URI",
	"mongodb://127.0.0.1:27017/" if _allow_local_mongo else None,
) or ""
DB_NAME = _secret_or_env(
	"DB_NAME",
	"face_verification" if _allow_local_mongo else None,
) or ""

# ----- Gmail Alerts ------------------------------------------
SENDER_EMAIL = _secret_or_env("SENDER_EMAIL", "")
SENDER_PASSWORD = _secret_or_env("SENDER_PASSWORD", "")
ADMIN_EMAIL = _secret_or_env("ADMIN_EMAIL", "")

# ----- Matching ----------------------------------------------
SIMILARITY_THRESHOLD = float(_secret_or_env("SIMILARITY_THRESHOLD", 0.75))
DUPLICATE_WINDOW_SEC = int(_secret_or_env("DUPLICATE_WINDOW_SEC", 60))
UNKNOWN_ALERT_COUNT = int(_secret_or_env("UNKNOWN_ALERT_COUNT", 3))
UNKNOWN_ALERT_WINDOW = int(_secret_or_env("UNKNOWN_ALERT_WINDOW", 5))

# ----- Model -------------------------------------------------
DEVICE = _secret_or_env("DEVICE", "cpu")
IMAGE_SIZE = int(_secret_or_env("IMAGE_SIZE", 160))

# ----- Dataset -----------------------------------------------
DATASET_PATH = _secret_or_env("DATASET_PATH", "data/processed_data")