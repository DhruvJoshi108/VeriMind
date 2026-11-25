"""Search backends and network-safe retrieval abstraction."""

from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from threading import Lock
from urllib.parse import quote, urlparse

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

    supports_query_variants = True

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return evidence results for a query."""
        raise NotImplementedError


class NullSearchBackend(SearchBackend):
    """Safe, no-network fallback backend."""

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return []


class TavilySearchBackend(SearchBackend):
    """Minimal real backend using the Tavily API if available."""

    supports_query_variants = False

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
        endpoint: str = "https://api.tavily.com/search",
    ) -> None:
        self.api_key = (
            api_key or os.getenv("SEARCH_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")
        )
        self.timeout = timeout
        self.endpoint = endpoint

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError(
                "SEARCH_API_KEY or WEB_SEARCH_API_KEY is required for Tavily web search."
            )
        if not query.strip() or k <= 0:
            return []
        response = requests.post(
            self.endpoint,
            json={
                "query": query.strip(),
                "max_results": min(k, 20),
                "api_key": self.api_key,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        search_results: list[SearchResult] = []
        for item in results[: min(k, 20)]:
            url = str(item.get("url") or "")
            text = str(item.get("content") or item.get("snippet") or item.get("title") or "")
            if not text.strip():
                continue
            search_results.append(
                SearchResult(
                    text=text,
                    url=url or None,
                    source=url or None,
                    domain=urlparse(url).hostname or "unknown",
                    score=max(0.0, min(1.0, float(item.get("score", 0.5)))),
                    published=item.get("published_date"),
                )
            )
        return search_results


class WikipediaSearchBackend(SearchBackend):
    """Search Wikipedia's public MediaWiki API without requiring an API key."""

    endpoint = "https://en.wikipedia.org/w/api.php"
    supports_query_variants = False

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._request_lock = Lock()
        self._last_request_at = 0.0
        self._blocked_until = 0.0

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not query.strip() or k <= 0:
            return []

        with self._request_lock:
            now = time.monotonic()
            wait_seconds = max(
                1.0 - (now - self._last_request_at),
                self._blocked_until - now,
                0.0,
            )
            if wait_seconds:
                time.sleep(wait_seconds)
            self._last_request_at = time.monotonic()
            response = requests.get(
                self.endpoint,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query.strip(),
                    "srnamespace": 0,
                    "srlimit": min(k, 10),
                    "srprop": "snippet|timestamp",
                    "format": "json",
                    "formatversion": 2,
                },
                headers={"User-Agent": "VERIMIND/0.2 (Wikipedia evidence retrieval)"},
                timeout=self.timeout,
            )
            if response.status_code == 429:
                try:
                    retry_after = float(response.headers.get("Retry-After", "5"))
                except ValueError:
                    retry_after = 5.0
                self._blocked_until = time.monotonic() + max(5.0, retry_after)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("query"), dict):
            raise ValueError("Wikipedia API response is missing query data")
        search_results = payload["query"].get("search")
        if not isinstance(search_results, list):
            raise ValueError("Wikipedia API response is missing query.search results")

        results: list[SearchResult] = []
        for item in search_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            snippet = html.unescape(re.sub(r"<[^>]+>", "", str(item.get("snippet", "")))).strip()
            if not title or not snippet:
                continue
            page_id = item.get("pageid")
            page_url = (
                f"https://en.wikipedia.org/?curid={page_id}"
                if page_id is not None
                else f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            )
            results.append(
                SearchResult(
                    text=f"{title}: {snippet}",
                    url=page_url,
                    source=page_url,
                    domain="wikipedia.org",
                    score=0.5,
                    published=item.get("timestamp"),
                )
            )
        return results


def get_search_backend() -> SearchBackend:
    """Return configured backend, defaulting to the null backend."""
    mode = settings.search_backend
    if mode == "tavily":
        return TavilySearchBackend(
            api_key=settings.search_api_key,
            endpoint=settings.web_search_api_url,
        )
    if mode == "wikipedia":
        return WikipediaSearchBackend()
    return NullSearchBackend()
