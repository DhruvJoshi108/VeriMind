"""Search backends and network-safe retrieval abstraction."""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from verimind.config import settings


@dataclass
class SearchResult:
    """One search result with metadata for ranking and auditing."""

    text: str
    url: str | None = None
    source: str | None = None
    domain: str = "unknown"
    score: float = 0.5
    published: str | None = None


class SearchBackend:
    """Abstract backend interface for evidence search."""

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return evidence results for a query."""
        raise NotImplementedError


class NullSearchBackend(SearchBackend):
    """Safe, no-network fallback backend."""

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return []


class TavilySearchBackend(SearchBackend):
    """Minimal real backend using the Tavily API if available."""

    def __init__(self, api_key: str | None = None, timeout: float = 10.0) -> None:
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
        self.timeout = timeout

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not self.api_key:
            return []
        response = requests.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": k, "api_key": self.api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        return [
            SearchResult(
                text=item.get("content") or item.get("snippet") or item.get("title") or "",
                url=item.get("url"),
                source=item.get("url"),
                domain=(item.get("url") or "unknown").split("/")[2] if (item.get("url") or "").startswith("http") else "unknown",
                score=0.7,
                published=item.get("published_date"),
            )
            for item in results[:k]
        ]


def get_search_backend() -> SearchBackend:
    """Return configured backend, defaulting to the null backend."""
    mode = settings.search_backend
    if mode == "tavily":
        return TavilySearchBackend(api_key=settings.search_api_key)
    return NullSearchBackend()
