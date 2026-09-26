"""
Dense / semantic retrieval.

Two backends are supported, chosen automatically:

1. sentence-transformers (if installed) — real neural embeddings, best
   quality, understands synonyms and paraphrase.
2. TF-IDF + Truncated SVD (LSA) — a lightweight, fully offline fallback
   built on scikit-learn. No model download required. Still captures
   co-occurrence / topical similarity, which is enough to demonstrate
   (and genuinely use) the hybrid-search pattern anywhere the real
   embedding model isn't available.

Both backends expose the same interface: fit(doc_ids, texts) and
score(query) -> {doc_id: cosine_similarity}.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class _LSABackend:
    """Offline semantic backend: TF-IDF -> truncated SVD (LSA)."""

    name = "tfidf-svd (offline)"

    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.svd: TruncatedSVD | None = None
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def fit(self, doc_ids: list[str], texts: list[str]) -> "_LSABackend":
        self.doc_ids = doc_ids
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, max(2, min(tfidf.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.doc_vectors = self.svd.fit_transform(tfidf)
        return self

    def score(self, query: str) -> dict[str, float]:
        q_tfidf = self.vectorizer.transform([query])
        q_vec = self.svd.transform(q_tfidf)
        sims = cosine_similarity(q_vec, self.doc_vectors)[0]
        return {doc_id: float(s) for doc_id, s in zip(self.doc_ids, sims)}


class _SentenceTransformerBackend:
    """Real neural embedding backend (used only if the package + model
    weights are available — this needs network access to download the
    model the first time)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def fit(self, doc_ids: list[str], texts: list[str]) -> "_SentenceTransformerBackend":
        self.doc_ids = doc_ids
        self.doc_vectors = self.model.encode(texts, normalize_embeddings=True)
        return self

    def score(self, query: str) -> dict[str, float]:
        q_vec = self.model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q_vec, self.doc_vectors)[0]
        return {doc_id: float(s) for doc_id, s in zip(self.doc_ids, sims)}


def build_semantic_backend(prefer_neural: bool = True):
    """Pick the best available semantic backend. Falls back gracefully."""
    if prefer_neural:
        try:
            return _SentenceTransformerBackend()
        except Exception:
            pass  # package missing or no network to fetch weights
    return _LSABackend()
