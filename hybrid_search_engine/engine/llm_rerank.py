"""
Optional LLM reranking stage.

Hybrid (BM25 + semantic) search gives you a fast, cheap candidate shortlist.
An LLM reranker then reads the query + each candidate's text and produces a
relevance judgment that's far more accurate than any vector similarity score,
at the cost of latency/money — so it's only run on the top-N candidates,
never the whole corpus.

This module is intentionally provider-agnostic: it calls whatever
`llm_call` function you pass in (query, document_text) -> float score in
[0, 1]. A ready-made `anthropic_llm_call` is included for the Anthropic API;
it only activates if the `anthropic` package is installed and an API key is
configured, and the engine works perfectly well without it (it just skips
the reranking stage).
"""
from __future__ import annotations

import json
from typing import Callable, Optional

LLMScoreFn = Callable[[str, str], float]


RERANK_PROMPT = """You are a search relevance judge.

Query: {query}

Document:
\"\"\"{document}\"\"\"

On a scale of 0.0 (irrelevant) to 1.0 (perfectly answers the query), how
relevant is this document to the query? Respond with ONLY a JSON object:
{{"score": <float>}}"""


def anthropic_llm_call(query: str, document: str, model: str = "claude-sonnet-4-6") -> Optional[float]:
    """Score one (query, document) pair with a Claude model.

    Returns None (caller should skip reranking) if the anthropic package
    or an API key isn't available, so the rest of the engine keeps working
    with pure BM25 + semantic scores.
    """
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": RERANK_PROMPT.format(query=query, document=document[:2000]),
            }],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        data = json.loads(text.strip().strip("`"))
        return max(0.0, min(1.0, float(data["score"])))
    except Exception:
        return None


def llm_rerank(
    query: str,
    candidates: list[tuple[str, str]],  # [(doc_id, doc_text), ...]
    llm_call: LLMScoreFn = anthropic_llm_call,
) -> dict[str, float]:
    """Score each candidate with the LLM. Missing/failed calls score 0.0
    and simply fall back to the underlying hybrid rank for that doc."""
    scores = {}
    for doc_id, text in candidates:
        s = llm_call(query, text)
        scores[doc_id] = s if s is not None else 0.0
    return scores
