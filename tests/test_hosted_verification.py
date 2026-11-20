from unittest.mock import Mock

import requests

from verimind.evidence_judge import GroqEvidenceJudge
from verimind.risk_scoring import RiskScorer


def test_hosted_judge_parses_evidence_bound_result(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"verdict":"supported","confidence":0.9,'
                        '"explanation":"Evidence states this.","citations":[1,99]}'
                    )
                }
            }
        ]
    }
    monkeypatch.setattr("verimind.evidence_judge.requests.post", Mock(return_value=response))
    judge = GroqEvidenceJudge(api_key="test-key")
    monkeypatch.setattr(type(judge), "enabled", property(lambda _self: True))

    result = judge.judge("Paris is in France.", [{"text": "Paris is in France.", "source": "source"}])

    assert result == {
        "verdict": "supported",
        "confidence": 0.9,
        "explanation": "Evidence states this.",
        "citations": [1],
        "provider": "groq",
        "model": judge.model,
    }


def test_hosted_judge_failure_falls_back_cleanly(monkeypatch):
    monkeypatch.setattr("verimind.evidence_judge.requests.post", Mock(side_effect=requests.Timeout("unavailable")))
    judge = GroqEvidenceJudge(api_key="test-key")
    monkeypatch.setattr(type(judge), "enabled", property(lambda _self: True))

    assert judge.judge("claim", [{"text": "evidence"}]) is None


def test_risk_scorer_uses_hosted_refutation():
    result = RiskScorer().score_claim(
        "Paris is in Germany.",
        [{"text": "Paris is in France.", "score": 0.9}],
        {"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1},
        {"verdict": "refuted", "confidence": 0.9, "explanation": "Evidence says France."},
    )

    assert result["verdict"] == "unsupported"
    assert result["risk_score"] == 96.5
    assert result["why"] == "Evidence says France."
