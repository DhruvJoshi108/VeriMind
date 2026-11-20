"""Local trusted knowledge base for offline evidence lookup."""

from __future__ import annotations

from typing import Any


class KnowledgeBase:
    """Very small trusted corpus used as a fallback evidence source."""

    def __init__(self) -> None:
        self.facts: list[dict[str, str]] = [
            {
                "text": "Paris is the capital city of France.",
                "source": "https://en.wikipedia.org/wiki/Paris",
                "domain": "wikipedia.org",
                "score": 0.95,
            },
            {
                "text": "The Eiffel Tower is in Paris, France.",
                "source": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                "domain": "wikipedia.org",
                "score": 0.92,
            },
            {
                "text": "France is a country in Europe.",
                "source": "https://en.wikipedia.org/wiki/France",
                "domain": "wikipedia.org",
                "score": 0.93,
            },
            {
                "text": "The capital of Germany is Berlin.",
                "source": "https://en.wikipedia.org/wiki/Berlin",
                "domain": "wikipedia.org",
                "score": 0.9,
            },
        ]

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return local facts with simple keyword matching."""
        normalized = query.lower()
        matches: list[dict[str, Any]] = []
        for fact in self.facts:
            fact_text = fact["text"].lower()
            if normalized in fact_text or any(token in fact_text for token in normalized.split() if len(token) > 3):
                matches.append(fact)
        return matches[:k]
