"""
Okapi BM25 — the classic keyword / lexical retrieval algorithm.

Implemented from scratch (numpy only) so the engine has zero heavyweight
or network-dependent dependencies for the lexical half of the hybrid search.
"""
from collections import Counter
import math

import numpy as np

from .utils import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_freqs: list[Counter] = []   # term counts per document
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.N: int = 0

    def fit(self, doc_ids: list[str], texts: list[str]) -> "BM25":
        self.doc_ids = doc_ids
        self.doc_freqs = []
        self.doc_lens = []
        df: Counter = Counter()

        for text in texts:
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            counts = Counter(tokens)
            self.doc_freqs.append(counts)
            for term in counts:
                df[term] += 1

        self.N = len(texts)
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0

        # standard BM25 idf, floored at a small positive value
        self.idf = {
            term: max(
                math.log((self.N - freq + 0.5) / (freq + 0.5) + 1),
                1e-9,
            )
            for term, freq in df.items()
        }
        return self

    def score(self, query: str) -> dict[str, float]:
        """Return {doc_id: bm25_score} for every document."""
        q_terms = tokenize(query)
        scores = np.zeros(self.N)

        for term in q_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, counts in enumerate(self.doc_freqs):
                f = counts.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom

        return {doc_id: float(s) for doc_id, s in zip(self.doc_ids, scores)}
