# categorization.py
from sentence_transformers import SentenceTransformer
import numpy as np

# ===============================
# LOAD MODEL (ONCE)
# ===============================
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

BASE_THRESHOLD = 0.35
UNCAT = "UNCATEGORIZED"


# ===============================
# COSINE SIMILARITY
# ===============================
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ===============================
# MAIN CATEGORIZATION FUNCTION
# ===============================
def categorize_emails(
    emails,
    label_centroids,
    user_categories,
    user_examples=None
):
    """
    emails: list of dicts (each with 'Body')
    label_centroids: frozen Gmail centroids
    user_categories: list of category names (strings)
    user_examples: dict {category: [example texts]}
    """

    known = [c for c in user_categories if c in label_centroids]
    unknown = [c for c in user_categories if c not in label_centroids]

    # -----------------------------
    # Build active centroids
    # -----------------------------
    active_centroids = {}

    # Known categories
    for c in known:
        active_centroids[c] = label_centroids[c]

    # Unknown categories → build from examples
    if user_examples:
        for cat, examples in user_examples.items():
            emb = embedder.encode(
                examples,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            active_centroids[cat] = emb.mean(axis=0)

    # -----------------------------
    # Predict
    # -----------------------------
    results = []

    for email in emails:
        text = email.get("Body", "")
        if not text.strip():
            email["Predicted_Category"] = UNCAT
            continue

        emb = embedder.encode(text, normalize_embeddings=True)

        best_label = UNCAT
        best_score = -1.0

        for lbl, centroid in active_centroids.items():
            score = cosine_sim(emb, centroid)
            if score > best_score:
                best_score = score
                best_label = lbl

        if best_score < BASE_THRESHOLD:
            best_label = UNCAT

        email["Predicted_Category"] = best_label
        email["Category_Score"] = round(best_score, 3)

        results.append(email)

    return results, unknown
