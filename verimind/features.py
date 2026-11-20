"""Feature engineering for claim-evidence scoring."""

from __future__ import annotations

import math
import re
from collections import Counter


def token_overlap(claim: str, evidence: str) -> float:
    """Return token-overlap similarity between claim and evidence."""
    claim_tokens = set(re.findall(r"\w+", claim.lower()))
    evidence_tokens = set(re.findall(r"\w+", evidence.lower()))
    if not claim_tokens or not evidence_tokens:
        return 0.0
    overlap = claim_tokens & evidence_tokens
    return len(overlap) / max(len(claim_tokens | evidence_tokens), 1)


def cosine_similarity(claim: str, evidence: str) -> float:
    """Compute a simple cosine similarity over bag-of-words counts."""
    claim_counter = Counter(re.findall(r"\w+", claim.lower()))
    evidence_counter = Counter(re.findall(r"\w+", evidence.lower()))
    numerator = sum(claim_counter[token] * evidence_counter[token] for token in set(claim_counter) | set(evidence_counter))
    claim_norm = math.sqrt(sum(value * value for value in claim_counter.values()))
    evidence_norm = math.sqrt(sum(value * value for value in evidence_counter.values()))
    if claim_norm == 0 or evidence_norm == 0:
        return 0.0
    return numerator / (claim_norm * evidence_norm)


def numeric_mismatch(claim: str, evidence: str) -> float:
    """Return a simple mismatch penalty for differing numbers."""
    claim_numbers = re.findall(r"\d+(?:\.\d+)?", claim)
    evidence_numbers = re.findall(r"\d+(?:\.\d+)?", evidence)
    if not claim_numbers or not evidence_numbers:
        return 0.0
    return 1.0 if set(claim_numbers) != set(evidence_numbers) else 0.0


def negation_mismatch(claim: str, evidence: str) -> float:
    """Return a basic penalty when claim and evidence differ on negation."""
    claim_neg = "not" in claim.lower() or "never" in claim.lower()
    evidence_neg = "not" in evidence.lower() or "never" in evidence.lower()
    if claim_neg != evidence_neg:
        return 1.0
    return 0.0


def entity_overlap(claim: str, evidence: str) -> float:
    """Return the proportion of capitalized entities shared by claim/evidence."""
    claim_entities = set(re.findall(r"\b[A-Z][a-z]+\b", claim))
    evidence_entities = set(re.findall(r"\b[A-Z][a-z]+\b", evidence))
    if not claim_entities and not evidence_entities:
        return 0.0
    return len(claim_entities & evidence_entities) / max(len(claim_entities | evidence_entities), 1)


def source_trust(domain: str) -> float:
    """Assign a score based on domain type."""
    trusted = ["gov", "edu", "wikipedia.org", "britannica.com", "nytimes.com", "reuters.com"]
    lower = domain.lower()
    if any(item in lower for item in trusted):
        return 0.9
    if lower.endswith(".org"):
        return 0.7
    if lower.endswith(".com"):
        return 0.5
    return 0.3


def build_features(claim: str, evidence: str, domain: str) -> dict[str, float]:
    """Compute a minimal feature dict used by the risk model."""
    return {
        "token_overlap": token_overlap(claim, evidence),
        "cosine_similarity": cosine_similarity(claim, evidence),
        "numeric_mismatch": numeric_mismatch(claim, evidence),
        "negation_mismatch": negation_mismatch(claim, evidence),
        "entity_overlap": entity_overlap(claim, evidence),
        "source_trust": source_trust(domain),
    }
