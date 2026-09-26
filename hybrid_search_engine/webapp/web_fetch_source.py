"""
Fetches live web results with no API key required, using DuckDuckGo.

The `ddgs` package (formerly `duckduckgo_search`) scrapes DuckDuckGo's
public search results — free, no signup, no key. This is what makes
"live web search" possible for a student project without paying for a
Google/Bing Search API key.
"""
from __future__ import annotations


def fetch_web_results(query: str, max_results: int = 15) -> list[dict]:
    """
    Returns a list of {"id": url, "title": str, "text": str, "url": url}
    where "text" is the title + snippet DuckDuckGo returns (used for
    BM25 / semantic scoring — we don't crawl the full page).
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError as e:
            raise RuntimeError(
                "Install the web-search backend with: pip install ddgs"
            ) from e

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("link") or ""
            title = r.get("title", "")
            snippet = r.get("body", "") or r.get("snippet", "")
            if not url:
                continue
            results.append({
                "id": url,
                "url": url,
                "title": title,
                "text": f"{title}. {snippet}",
                "snippet": snippet,
            })
    return results
