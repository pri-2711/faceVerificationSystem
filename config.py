"""Application settings with cloud-safe fallbacks.

Values are loaded in this order:
1. Streamlit secrets
2. Environment variables
3. Local defaults
"""

from __future__ import annotations

import os


def _setting(name: str, default):
	try:
		import streamlit as st
		if name in st.secrets:
			return st.secrets[name]
	except Exception:
		pass

	return os.getenv(name, default)


# ----- MongoDB -----------------------------------------------
MONGO_URI = _setting("MONGO_URI", "mongodb://127.0.0.1:27017/")
DB_NAME = _setting("DB_NAME", "face_verification")

# ----- Gmail Alerts ------------------------------------------
SENDER_EMAIL = _setting("SENDER_EMAIL", "")
SENDER_PASSWORD = _setting("SENDER_PASSWORD", "")
ADMIN_EMAIL = _setting("ADMIN_EMAIL", "")

# ----- Matching ----------------------------------------------
SIMILARITY_THRESHOLD = float(_setting("SIMILARITY_THRESHOLD", 0.75))
DUPLICATE_WINDOW_SEC = int(_setting("DUPLICATE_WINDOW_SEC", 60))
UNKNOWN_ALERT_COUNT = int(_setting("UNKNOWN_ALERT_COUNT", 3))
UNKNOWN_ALERT_WINDOW = int(_setting("UNKNOWN_ALERT_WINDOW", 5))

# ----- Model -------------------------------------------------
DEVICE = _setting("DEVICE", "cpu")
IMAGE_SIZE = int(_setting("IMAGE_SIZE", 160))

# ----- Dataset -----------------------------------------------
DATASET_PATH = _setting("DATASET_PATH", "data/processed_data")