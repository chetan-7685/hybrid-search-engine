# Hybrid Search Engine (Python + LLMs)

A search engine that fuses two complementary retrieval strategies, plus an
optional LLM reranking stage:

1. **Lexical search (BM25)** — exact keyword matching. Great at finding rare,
   specific terms (`"BM25"`, an error code, a product SKU) that vector search
   can blur past.
2. **Semantic search (embeddings)** — dense vector similarity. Great at
   matching *meaning*: a query like `"model that learns from data like the
   brain"` finds a document about neural networks even though it shares no
   words with it.
3. **LLM reranking (optional)** — once the two retrievers above produce a
   shortlist, an LLM reads the query and each candidate and gives a real
   relevance judgment — the most accurate signal, used only on a small
   top-N pool since it's the slowest/most expensive step.

Scores from BM25 and the embedding model are min-max normalized and combined
with a tunable weight `alpha` (`0` = pure keyword, `1` = pure semantic,
`0.5` = balanced).

## Project layout

```
hybrid_search_engine/
├── engine/
│   ├── bm25.py         # BM25 implemented from scratch (numpy only)
│   ├── semantic.py     # embedding backend: sentence-transformers if
│   │                   #   available, else an offline TF-IDF+SVD fallback
│   ├── llm_rerank.py   # optional LLM reranking stage (Claude API)
│   ├── hybrid.py        # fuses the two retrievers + orchestrates reranking
│   └── utils.py         # tokenization, score normalization
├── webapp/
│   ├── app.py                # Flask server: /search fetches live web
│   │                          #   results and reranks them with `engine`
│   ├── web_fetch_source.py   # fetches live results via DuckDuckGo (no API key)
│   ├── templates/index.html  # search box + results UI
│   └── static/style.css
├── demo.py              # terminal demo over a small built-in sample corpus
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

`numpy` and `scikit-learn` are required. `sentence-transformers` and
`anthropic` are optional — the engine auto-detects and falls back gracefully
if they aren't installed (offline TF-IDF+SVD embeddings, no LLM reranking).

## Quickstart — terminal demo

```bash
python demo.py
python demo.py --alpha 0.8              # weight semantic matches higher
python demo.py --query "your own query here"
python demo.py --llm-rerank             # needs `pip install anthropic` + ANTHROPIC_API_KEY
```

## Quickstart — web app (search box, live web results)

This is the version to demo in class: a browser search box that fetches
live results from the web (via DuckDuckGo — free, no API key) and reranks
them with the hybrid engine in real time.

```bash
pip install -r requirements.txt
python webapp/app.py
```

Then open **http://127.0.0.1:5000** in your browser. Type a query, hit
search, and drag the keyword↔semantic slider to see the ranking change live.

Each result card shows the fused score plus the individual BM25 and
semantic scores — that's the part worth pointing at during a viva, since it
visually proves both retrieval methods are contributing.

## Use it in your own code

```python
from engine import HybridSearchEngine

docs = [
    {"id": "1", "text": "Python lists are ordered, mutable collections."},
    {"id": "2", "text": "A neural network learns patterns from data."},
    # ... any extra keys (title, url, category...) are kept as metadata
]

engine = HybridSearchEngine(alpha=0.5)
engine.index(docs)

results = engine.search("machine learning model", top_k=5)
for r in results:
    print(r.doc_id, r.score, r.text)
```

## How the pieces fit together

```
query
  │
  ├──► BM25.score()        → {doc_id: lexical_score}
  ├──► semantic.score()    → {doc_id: cosine_similarity}
  │
  ▼
normalize both to [0,1], fuse:
  fused = alpha * semantic + (1 - alpha) * bm25
  │
  ▼
top-N candidates
  │
  └──► (optional) LLM reranker reads query + each candidate
        and outputs a relevance score → final ranking
```

## Swapping in real neural embeddings

By default the semantic backend tries `sentence-transformers`
(`all-MiniLM-L6-v2`) first. If it's not installed, or the model weights
can't be downloaded (no network), it automatically falls back to a
TF-IDF + truncated SVD (LSA) backend built on scikit-learn — fully offline,
no model download, still captures topical/co-occurrence similarity well
enough to demonstrate (and genuinely use) the hybrid pattern.

Install the neural backend for better semantic quality:

```bash
pip install sentence-transformers
```

## Swapping in a different LLM for reranking

`engine/llm_rerank.py` is provider-agnostic: pass any function
`(query: str, document: str) -> float` as `llm_call` to
`engine.search(..., rerank_with_llm=True, llm_call=my_fn)`. A ready-made
`anthropic_llm_call` is included for the Claude API.

## Extending this to a production-scale index

This demo scores every document per query (fine for thousands of docs). To
scale further:
- Swap the semantic backend for an approximate nearest-neighbor index
  (FAISS, HNSW, or a vector DB like Pinecone/Weaviate/Qdrant).
- Swap BM25 for a real inverted-index engine (Elasticsearch/OpenSearch,
  or `rank_bm25`/Lucene) so you don't rescore the whole corpus per query.
- Keep the same fusion + LLM-reranking pattern on top — it's index-agnostic.
