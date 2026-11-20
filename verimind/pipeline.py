"""Main verification pipeline orchestrating claims, evidence, scoring, and reporting."""

from __future__ import annotations

import time
from typing import Any

from verimind.claim_segmentation import split_claims
from verimind.data import build_synthetic_dataset
from verimind.evidence_judge import GroqEvidenceJudge
from verimind.features import cosine_similarity, token_overlap
from verimind.nli_model import NLIModel
from verimind.retrieval import HybridRetriever
from verimind.risk_scoring import RiskScorer


class VerificationPipeline:
    """End-to-end verification pipeline for a document or answer."""

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.risk_scorer = RiskScorer()
        self.judge = GroqEvidenceJudge()
        self.nli_model = self._train_default_model()

    def _train_default_model(self) -> NLIModel:
        """Train the deterministic offline baseline used when no artifact is configured."""
        dataset = build_synthetic_dataset()
        return NLIModel().fit(dataset["pairs"], dataset["labels"])

    def _classify_claim(self, claim: str, evidence: list[dict[str, Any]]) -> dict[str, float]:
        if not evidence:
            return {"entailment": 0.15, "contradiction": 0.20, "neutral": 0.65}

        relevant_evidence = [
            item
            for item in evidence
            if max(
                token_overlap(claim, str(item.get("text", ""))),
                cosine_similarity(claim, str(item.get("text", ""))),
            )
            >= 0.12
        ]
        if not relevant_evidence:
            return {"entailment": 0.10, "contradiction": 0.10, "neutral": 0.80}

        pairs = [(claim, str(item["text"])) for item in relevant_evidence]
        probabilities = self.nli_model.predict_proba(pairs)
        rows = [
            {label: float(value) for label, value in zip(self.nli_model.classes_, row, strict=True)}
            for row in probabilities
        ]
        # Evidence is disjunctive: a single strong source should not be diluted by
        # weakly related retrieved passages.
        entailment = max(row.get("entailment", 0.0) for row in rows)
        contradiction = max(row.get("contradiction", 0.0) for row in rows)
        neutral = max(0.0, 1.0 - max(entailment, contradiction))
        total = entailment + contradiction + neutral
        return {
            "entailment": entailment / total,
            "contradiction": contradiction / total,
            "neutral": neutral / total,
        }

    def verify_text(self, text: str) -> dict[str, Any]:
        """Verify a document and return per-claim and overall report."""
        claims = split_claims(text)
        start = time.perf_counter()
        claim_entries: list[dict[str, Any]] = []
        for claim in claims:
            claim_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            evidence = self.retriever.search(claim, k=3)
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
            nli_probs = self._classify_claim(claim, evidence)
            judge_start = time.perf_counter()
            judge_result = self.judge.judge(claim, evidence)
            judge_latency_ms = (
                (time.perf_counter() - judge_start) * 1000
                if self.judge.enabled and evidence
                else None
            )
            scored = self.risk_scorer.score_claim(claim, evidence, nli_probs, judge_result)
            claim_entries.append(
                {
                    "claim": claim,
                    "verification_latency_ms": round((time.perf_counter() - claim_start) * 1000, 2),
                    "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                    "judge_latency_ms": (
                        round(judge_latency_ms, 2) if judge_latency_ms is not None else None
                    ),
                    "judge_error": bool(self.judge.enabled and evidence and judge_result is None),
                    "evidence": evidence,
                    "nli_probs": nli_probs,
                    "judge": judge_result,
                    "risk_score": scored["risk_score"],
                    "verdict": scored["verdict"],
                    "why": scored["why"],
                }
            )

        document_summary = {
            "total_claims": len(claims),
            "flagged_claims": sum(1 for item in claim_entries if item["risk_score"] >= 60),
            "overall_risk_score": round(sum(item["risk_score"] for item in claim_entries) / max(1, len(claim_entries)), 2),
            "total_latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
        return {
            "document_summary": document_summary,
            "claims": claim_entries,
        }
