"""Explainable risk scoring for claim-level verification."""

from __future__ import annotations

from typing import Any


class RiskScorer:
    """Convert NLI probabilities and evidence confidence into a risk score."""

    def score_claim(
        self,
        claim: str,
        evidence: list[dict[str, Any]],
        nli_probs: dict[str, float],
        judge_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score one claim and provide a verdict plus explanation."""
        entailment = float(nli_probs.get("entailment", 0.0))
        contradiction = float(nli_probs.get("contradiction", 0.0))
        neutral = float(nli_probs.get("neutral", 0.0))

        retrieval_score = 0.0
        if evidence:
            retrieval_score = sum(float(item.get("score", 0.0)) for item in evidence) / len(evidence)

        exact_support = 0.0
        claim_text = claim.strip().lower()
        if evidence:
            exact_support = max(
                1.0 if claim_text in str(item.get("text", "")).strip().lower() else 0.0
                for item in evidence
            )

        support_score = entailment * 0.6 + retrieval_score * 0.4 + exact_support * 0.5
        support_score = max(0.0, min(1.0, support_score))

        contradiction_score = contradiction * 0.7 + (1.0 - retrieval_score) * 0.3
        uncertainty = neutral * 0.6 + (1.0 - retrieval_score) * 0.4

        risk_score = max(
            0.0,
            min(
                100.0,
                (1.0 - support_score + contradiction_score * 0.5 + uncertainty * 0.25) * 100.0,
            ),
        )

        if judge_result is not None:
            judge_verdict = judge_result["verdict"]
            confidence = float(judge_result["confidence"])
            verdict = {
                "supported": "supported",
                "refuted": "unsupported",
                "insufficient_evidence": "uncertain",
            }[judge_verdict]
            why = judge_result["explanation"] or "The hosted judge could not provide an explanation."
            risk_score = {
                "supported": (1.0 - confidence) * 35.0,
                "refuted": 65.0 + confidence * 35.0,
                "insufficient_evidence": 55.0 + (1.0 - confidence) * 20.0,
            }[judge_verdict]
        elif support_score >= 0.75 and contradiction_score < 0.3:
            verdict = "supported"
            why = "The claim is strongly supported by directly relevant evidence."
        elif contradiction_score >= 0.45:
            verdict = "unsupported"
            why = "The claim conflicts with the strongest available evidence."
        else:
            verdict = "uncertain"
            why = "The claim is weakly supported or lacks enough trustworthy evidence."

        return {
            "claim": claim,
            "risk_score": round(risk_score, 2),
            "verdict": verdict,
            "why": why,
            "evidence_count": len(evidence),
        }
