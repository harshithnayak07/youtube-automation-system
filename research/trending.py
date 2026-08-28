"""Research module — trend discovery and topic selection."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Topic:
    """A single researched trending topic."""
    title: str
    url: str
    summary: str
    source: str


# ---------------------------------------------------------------------------
# Search provider protocol (swappable)
# ---------------------------------------------------------------------------

@runtime_checkable
class SearchProvider(Protocol):
    """Interface for pluggable search backends."""
    def search(self, query: str, max_results: int = 10) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Default provider — Google News RSS (no API key required)
# ---------------------------------------------------------------------------

_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

class GoogleNewsSearchProvider:
    """Fetches recent articles via Google News RSS feed."""

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        resp = requests.get(_GOOGLE_NEWS_RSS, params=params, timeout=15)
        resp.raise_for_status()

        items: list[dict] = []
        # Simple XML parsing without pulling in lxml
        for match in _parse_rss_items(resp.text):
            items.append(match)
            if len(items) >= max_results:
                break
        return items


def _parse_rss_items(xml_text: str) -> list[dict]:
    """Minimal RSS item extraction — avoids extra dependencies."""
    results: list[dict] = []
    parts = xml_text.split("<item>")[1:]  # skip header
    for block in parts:
        title = _extract_tag(block, "title")
        link = _extract_tag(block, "link")
        source = _extract_tag(block, "source")
        if title and link:
            results.append({
                "title": title,
                "url": link,
                "source": source or "Google News",
            })
    return results


def _extract_tag(text: str, tag: str) -> str | None:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = text.find(open_tag)
    if start == -1:
        return None
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return None
    return text[start:end].strip()


# ---------------------------------------------------------------------------
# Topic selection
# ---------------------------------------------------------------------------

_AI_QUERIES = [
    "artificial intelligence breaking news today",
    "new AI model release 2026",
    "AI industry major announcement",
    "generative AI latest development",
    "AI startup funding news",
]


def find_trending_ai_topic(
    search_provider: SearchProvider | None = None,
    *,
    queries: list[str] | None = None,
) -> Topic:
    """Search for a current AI-related trending topic and return one.

    Parameters
    ----------
    search_provider:
        Pluggable search backend.  Defaults to GoogleNewsSearchProvider.
    queries:
        Override the built-in query list.

    Returns
    -------
    Topic
        A single selected topic with title, url, summary, and source.

    Raises
    ------
    RuntimeError
        If no topics could be retrieved from any query.
    """
    provider = search_provider or GoogleNewsSearchProvider()
    candidate_queries = queries or _AI_QUERIES

    all_results: list[dict] = []
    for query in candidate_queries:
        try:
            results = provider.search(query, max_results=5)
            all_results.extend(results)
            logger.debug("Query %r returned %d results", query, len(results))
        except Exception:
            logger.warning("Search failed for query %r", query, exc_info=True)

    if not all_results:
        raise RuntimeError(
            "No trending AI topics found — all search queries returned empty"
        )

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for item in all_results:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            unique.append(item)

    selected = random.choice(unique)
    logger.info("Selected topic: %s", selected["title"])

    return Topic(
        title=selected["title"],
        url=selected["url"],
        summary=f"Trending AI topic: {selected['title']}",
        source=selected.get("source", "Google News"),
    )
