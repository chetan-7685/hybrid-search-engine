"""
Demo: build a hybrid search engine over a small sample corpus and run a
few example queries, printing lexical, semantic, and fused scores side
by side so you can see what each half of the engine contributes.

Run:
    python demo.py
    python demo.py --alpha 0.7          # weight semantic search higher
    python demo.py --query "your text"  # run one custom query
    python demo.py --llm-rerank         # also rerank top results with an LLM
                                         # (requires `pip install anthropic`
                                         # and an ANTHROPIC_API_KEY env var)
"""
import argparse

from engine import HybridSearchEngine

CORPUS = [
    {"id": "doc1", "title": "Python Lists", "text": (
        "Python lists are ordered, mutable collections that can hold items "
        "of different types. You can append, remove, sort, and slice them."
    )},
    {"id": "doc2", "title": "Neural Networks Intro", "text": (
        "A neural network is a machine learning model inspired by the human "
        "brain, made of layers of interconnected nodes that learn patterns "
        "from data through backpropagation."
    )},
    {"id": "doc3", "title": "BM25 Ranking", "text": (
        "BM25 is a bag-of-words retrieval function that ranks documents "
        "based on term frequency and inverse document frequency, widely "
        "used in search engines like Elasticsearch."
    )},
    {"id": "doc4", "title": "Vector Embeddings", "text": (
        "Embeddings map words, sentences, or documents into dense numerical "
        "vectors such that semantically similar items end up close together "
        "in vector space, enabling similarity search."
    )},
    {"id": "doc5", "title": "Cooking Pasta", "text": (
        "Boil salted water, add pasta, cook until al dente, then toss with "
        "olive oil, garlic, and parmesan for a simple weeknight dinner."
    )},
    {"id": "doc6", "title": "Transformer Architecture", "text": (
        "Transformers use self-attention to weigh the relevance of every "
        "token against every other token, which made them the foundation "
        "of modern large language models like GPT and Claude."
    )},
    {"id": "doc7", "title": "Database Indexing", "text": (
        "A database index is a data structure, often a B-tree, that speeds "
        "up lookups on a table at the cost of extra storage and slower "
        "writes."
    )},
    {"id": "doc8", "title": "Climate Change Basics", "text": (
        "Rising greenhouse gas concentrations trap more heat in the "
        "atmosphere, driving global temperature increases, sea level rise, "
        "and more extreme weather events."
    )},
    {"id": "doc9", "title": "REST APIs", "text": (
        "REST is an architectural style for web APIs using standard HTTP "
        "verbs like GET, POST, PUT, and DELETE to operate on resources "
        "identified by URLs."
    )},
    {"id": "doc10", "title": "Recommender Systems", "text": (
        "Recommender systems suggest items to users based on collaborative "
        "filtering, content-based similarity, or hybrid approaches that "
        "combine both signals."
    )},
    {"id": "doc11", "title": "Hybrid Search", "text": (
        "Hybrid search combines sparse lexical retrieval like BM25 with "
        "dense semantic vector search, often fused with reciprocal rank "
        "fusion or a weighted score, to get both precision and recall."
    )},
    {"id": "doc12", "title": "Python Web Frameworks", "text": (
        "Flask and FastAPI are lightweight Python web frameworks for "
        "building APIs, while Django is a full-featured framework with an "
        "ORM and admin panel built in."
    )},
]


def print_result(rank, r):
    llm = f"  llm={r.llm_score:.2f}" if r.llm_score is not None else ""
    print(
        f"  {rank}. [{r.doc_id}] score={r.score:.3f}"
        f"  (bm25={r.bm25_score:.2f}  semantic={r.semantic_score:.2f}{llm})"
    )
    title = r.metadata.get("title", "")
    snippet = r.text if len(r.text) <= 100 else r.text[:97] + "..."
    print(f"     {title}: {snippet}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="0=pure keyword BM25, 1=pure semantic, 0.5=balanced")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--query", type=str, default=None,
                         help="Run a single custom query instead of the demo set")
    parser.add_argument("--llm-rerank", action="store_true",
                         help="Rerank the top candidates with an LLM (needs anthropic + API key)")
    args = parser.parse_args()

    engine = HybridSearchEngine(alpha=args.alpha)
    engine.index(CORPUS)
    print(f"Indexed {len(CORPUS)} documents. Semantic backend: {engine.semantic.name}. alpha={args.alpha}\n")

    queries = [args.query] if args.query else [
        "how do search engines rank documents by keyword",
        "model that learns from data like the brain",
        "combining keyword and vector search",
        "how to make dinner with noodles",
    ]

    for q in queries:
        print(f'Query: "{q}"')
        results = engine.search(q, top_k=args.top_k, rerank_with_llm=args.llm_rerank)
        for i, r in enumerate(results, 1):
            print_result(i, r)
        print()


if __name__ == "__main__":
    main()
