"""Optional web-search fallback interface."""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def search_web(query: str, k: int = 5) -> list[dict]:
    if not isinstance(query, str) or not query.strip() or k <= 0:
        return []
    key = os.getenv("WEB_SEARCH_API_KEY")
    endpoint = os.getenv("WEB_SEARCH_API_URL", "https://api.tavily.com/search")
    if not key:
        return []
    try:
        import requests
    except ImportError:
        return []
    try:
        response = requests.post(
            endpoint,
            json={"query": query, "max_results": k, "api_key": key},
            timeout=10,
        )
        response.raise_for_status()
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", item.get("snippet", "")),
            }
            for item in response.json().get("results", [])[:k]
        ]
    except (requests.RequestException, ValueError, TypeError):
        return []
