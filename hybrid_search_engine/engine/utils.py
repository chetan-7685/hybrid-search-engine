import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "these", "those",
    "or", "but", "if", "than", "so", "such", "can", "could", "should",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a {doc_id: score} dict to the [0, 1] range."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
