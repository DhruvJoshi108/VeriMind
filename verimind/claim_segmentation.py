"""Claim extraction and segmentation utilities."""

from __future__ import annotations

import re


def split_claims(text: str) -> list[str]:
    """Split an LLM response into atomic factual claims."""
    if not text or not text.strip():
        return []

    normalized = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    claims: list[str] = []
    for sentence in sentences:
        cleaned = sentence.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered.startswith(("hello", "hi ", "thanks", "thank you", "sure")):
            continue
        if lowered.endswith("?"):
            continue
        if len(cleaned.split()) < 3:
            continue
        if any(token in lowered for token in ["i think", "in my opinion", "i believe", "maybe", "possibly"]):
            continue
        claims.append(cleaned)
    return claims
