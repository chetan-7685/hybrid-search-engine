"""
HybridSearchEngine — combines:

  1. BM25            (exact keyword / lexical matching, fast, precise on rare terms)
  2. Semantic vectors (catches synonyms, paraphrase, conceptual matches)
  3. Optional LLM reranking of the top-N fused candidates (highest quality,
     highest cost — used sparingly)

Fusion strategy: normalize each score list to [0, 1], then combine with a
tunable weight `alpha` (weight on semantic score; 1 - alpha on BM25).
alpha=1.0 -> pure semantic, alpha=0.0 -> pure BM25, alpha=0.5 -> balanced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25
from .semantic import build_semantic_backend
from .llm_rerank import llm_rerank, LLMScoreFn
from .utils import normalize_scores


@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float
    bm25_score: float
    semantic_score: float
    llm_score: float | None = None
    metadata: dict = field(default_factory=dict)


class HybridSearchEngine:
    def __init__(self, alpha: float = 0.5, prefer_neural_embeddings: bool = True):
        """
        alpha: blend weight for semantic vs. lexical score (0..1).
        prefer_neural_embeddings: try sentence-transformers first, fall
            back to the offline TF-IDF+SVD backend if unavailable.
        """
        self.alpha = alpha
        self.bm25 = BM25()
        self.semantic = build_semantic_backend(prefer_neural=prefer_neural_embeddings)
        self._docs: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}
        self._fitted = False

    def index(self, documents: list[dict]) -> "HybridSearchEngine":
        """
        documents: list of {"id": str, "text": str, **any extra metadata}
        """
        doc_ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        self._docs = {d["id"]: d["text"] for d in documents}
        self._metadata = {
            d["id"]: {k: v for k, v in d.items() if k not in ("id", "text")}
            for d in documents
        }

        self.bm25.fit(doc_ids, texts)
        self.semantic.fit(doc_ids, texts)
        self._fitted = True
        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_with_llm: bool = False,
        llm_call: LLMScoreFn | None = None,
        rerank_pool: int = 10,
    ) -> list[SearchResult]:
        if not self._fitted:
            raise RuntimeError("Call .index(documents) before .search().")

        bm25_raw = self.bm25.score(query)
        sem_raw = self.semantic.score(query)

        bm25_norm = normalize_scores(bm25_raw)
        sem_norm = normalize_scores(sem_raw)

        fused = {
            doc_id: self.alpha * sem_norm.get(doc_id, 0.0)
            + (1 - self.alpha) * bm25_norm.get(doc_id, 0.0)
            for doc_id in self._docs
        }
        ranked_ids = sorted(fused, key=fused.get, reverse=True)

        llm_scores: dict[str, float] = {}
        if rerank_with_llm:
            pool_ids = ranked_ids[:rerank_pool]
            candidates = [(doc_id, self._docs[doc_id]) for doc_id in pool_ids]
            kwargs = {"llm_call": llm_call} if llm_call else {}
            llm_scores = llm_rerank(query, candidates, **kwargs)
            # LLM score becomes the primary sort key for the reranked pool,
            # with the fused hybrid score as a tiebreaker.
            ranked_ids = sorted(
                pool_ids,
                key=lambda d: (llm_scores.get(d, 0.0), fused.get(d, 0.0)),
                reverse=True,
            ) + [d for d in ranked_ids if d not in pool_ids]

        results = []
        for doc_id in ranked_ids[:top_k]:
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    text=self._docs[doc_id],
                    score=llm_scores.get(doc_id, fused[doc_id]),
                    bm25_score=bm25_norm.get(doc_id, 0.0),
                    semantic_score=sem_norm.get(doc_id, 0.0),
                    llm_score=llm_scores.get(doc_id) if rerank_with_llm else None,
                    metadata=self._metadata.get(doc_id, {}),
                )
            )
        return results
