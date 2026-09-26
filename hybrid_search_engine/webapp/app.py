"""
Flask web app: type a query into a search box in the browser, it fetches
live results from the web (DuckDuckGo, no API key), then reranks them with
the hybrid BM25 + semantic engine, and displays everything as cards —
no terminal interaction needed after you start the server.

Run:
    python webapp/app.py
Then open:
    http://127.0.0.1:5000
"""
import os
import sys

from flask import Flask, jsonify, render_template, request

# Make the sibling `engine/` package importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import HybridSearchEngine
from webapp.web_fetch_source import fetch_web_results

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    alpha = float(request.args.get("alpha", 0.5))

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        web_results = fetch_web_results(query, max_results=15)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if not web_results:
        return jsonify({"query": query, "results": [], "backend": None})

    engine = HybridSearchEngine(alpha=alpha)
    engine.index(web_results)
    ranked = engine.search(query, top_k=10)

    return jsonify({
        "query": query,
        "backend": engine.semantic.name,
        "alpha": alpha,
        "results": [
            {
                "url": r.doc_id,
                "title": r.metadata.get("title", r.doc_id),
                "snippet": r.metadata.get("snippet", ""),
                "score": round(r.score, 3),
                "bm25_score": round(r.bm25_score, 3),
                "semantic_score": round(r.semantic_score, 3),
            }
            for r in ranked
        ],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
