# =============================================================
#  similarity.py  —  Cosine similarity + matching logic
#
#  Imported by verify.py.
#  Kept separate so the matching logic can be tested or
#  swapped independently of the rest of the pipeline.
# =============================================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config


def compute_similarity(embedding_a, embedding_b):
    """
    Compute cosine similarity between two 512-dim embeddings.

    Returns:
        float in range [-1, 1]  (higher = more similar)
    """
    a = embedding_a.reshape(1, -1)
    b = embedding_b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


def find_best_match(query_embedding, registered_users):
    """
    Compare a query embedding against all registered users.
    Returns the best matching user if score >= threshold, else Unknown.

    Args:
        query_embedding  (np.ndarray)  : 512-dim vector from webcam/image
        registered_users (list of dict): each dict has {set_id, name, embedding}

    Returns:
        name   (str)   : matched name or "Unknown"
        set_id (str)   : matched set_id or None
        score  (float) : best cosine similarity score
    """
    if not registered_users:
        return "Unknown", None, 0.0

    best_score = -1.0
    best_user  = None

    for user in registered_users:
        score = compute_similarity(query_embedding, user["embedding"])
        if score > best_score:
            best_score = score
            best_user  = user

    if best_score >= config.SIMILARITY_THRESHOLD:
        return best_user["name"], best_user["set_id"], best_score

    return "Unknown", None, best_score


def normalize(embedding):
    """L2-normalize an embedding vector."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm