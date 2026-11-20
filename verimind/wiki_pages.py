"""Helpers for converting FEVER Wikipedia pages into retrieval passages."""

from __future__ import annotations

import re
from typing import Any

PASSAGE_WORDS = 200
PAGE_EXTRACT_CHARACTER_LIMIT = 40000
TOKEN_PATTERN = re.compile(r"\S+")


def page_passages(page: dict[str, Any]) -> list[dict[str, Any]]:
    page_id = str(page.get("id") or page.get("title") or "").strip()
    title = page_id.replace("_", " ")
    raw_text = page.get("text") or page.get("lines") or ""
    text = "\n".join(map(str, raw_text)) if isinstance(raw_text, list) else str(raw_text)
    text = text.strip()[:PAGE_EXTRACT_CHARACTER_LIMIT]
    if not title or not text:
        return []

    words = TOKEN_PATTERN.findall(text)
    result = []
    url_title = page_id.replace(" ", "_")
    for passage_id, start in enumerate(range(0, len(words), PASSAGE_WORDS)):
        passage = " ".join(words[start : start + PASSAGE_WORDS]).strip()
        if passage:
            result.append(
                {
                    "doc_id": url_title,
                    "title": title,
                    "passage_id": passage_id,
                    "source": f"https://en.wikipedia.org/wiki/{url_title}",
                    "domain": "wikipedia.org",
                    "text": passage,
                }
            )
    return result
