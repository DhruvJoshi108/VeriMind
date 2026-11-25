import logging
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

from api.main import app
from verimind.claim_segmentation import split_claims
from verimind.config import Settings
from verimind.knowledge_base import KnowledgeBase
from verimind.nli_model import NLIModel
from verimind.pipeline import VerificationPipeline
from verimind.retrieval import HybridRetriever
from verimind.risk_scoring import RiskScorer
from verimind.web_search import (
    SearchBackend,
    SearchResult,
    TavilySearchBackend,
    WikipediaSearchBackend,
)


def test_package_exports_are_importable():
    from verimind import (
        KnowledgeBase,
        LocalEvidenceRetriever,
        NLIModel,
        RiskScorer,
        VerificationPipeline,
        split_claims,
    )

    assert all(
        exported
        for exported in (
            KnowledgeBase,
            LocalEvidenceRetriever,
            NLIModel,
            RiskScorer,
            VerificationPipeline,
            split_claims,
        )
    )


def test_split_claims_filters_noise():
    text = "Hello there! Paris is the capital of France. The Eiffel Tower is in Paris. Thanks!"
    claims = split_claims(text)
    assert len(claims) >= 2
    assert all("hello" not in claim.lower() for claim in claims)


def test_knowledge_base_contains_fact():
    kb = KnowledgeBase()
    result = kb.search("Paris is the capital of France")
    assert result
    assert any("Paris" in item["text"] for item in result)


def test_nli_model_trains_and_predicts():
    model = NLIModel()
    X = [
        ("Paris is the capital of France.", "Paris is the capital of France."),
        ("Paris is the capital of France.", "Paris is the capital of Germany."),
        ("Paris is the capital of France.", "The Eiffel Tower is in Paris."),
    ]
    y = ["entailment", "contradiction", "neutral"]
    model.fit(X, y)
    prediction = model.predict_proba([X[0]])
    assert prediction.shape[0] == 1
    assert set(model.classes_) <= {"entailment", "contradiction", "neutral"}


def test_nli_model_trains_by_epoch_and_reports_validation_accuracy():
    from verimind.data import build_synthetic_dataset

    dataset = build_synthetic_dataset()
    validation_indices = {
        next(index for index, label in enumerate(dataset["labels"]) if label == target)
        for target in {"entailment", "contradiction", "neutral"}
    }
    train_pairs = [
        pair for index, pair in enumerate(dataset["pairs"]) if index not in validation_indices
    ]
    train_labels = [
        label for index, label in enumerate(dataset["labels"]) if index not in validation_indices
    ]
    validation_pairs = [dataset["pairs"][index] for index in sorted(validation_indices)]
    validation_labels = [dataset["labels"][index] for index in sorted(validation_indices)]

    history = NLIModel().fit_with_validation(
        train_pairs,
        train_labels,
        validation_pairs,
        validation_labels,
        epochs=3,
        patience=2,
    )

    assert 1 <= len(history) <= 3
    assert history[0]["epoch"] == 1
    assert all(0 <= row["validation_accuracy"] <= 1 for row in history)


def test_risk_scorer_outputs_valid_report():
    scorer = RiskScorer()
    result = scorer.score_claim(
        claim="Paris is the capital of France.",
        evidence=[{"text": "Paris is the capital city of France.", "score": 0.9, "source": "https://example.com"}],
        nli_probs={"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05},
    )
    assert 0 <= result["risk_score"] <= 100
    assert result["verdict"] in {"supported", "uncertain", "unsupported"}


def test_weak_retrieval_does_not_turn_moderate_contradiction_into_refutation():
    result = RiskScorer().score_claim(
        claim="The claim is true.",
        evidence=[{"text": "Weakly related passage.", "score": 0.0}],
        nli_probs={"entailment": 0.05, "contradiction": 0.6, "neutral": 0.35},
    )

    assert result["verdict"] == "uncertain"


def test_strong_contradiction_with_relevant_evidence_remains_unsupported():
    result = RiskScorer().score_claim(
        claim="Paris is the capital of Germany.",
        evidence=[
            {
                "text": "Paris is the capital of France.",
                "score": 0.9,
                "source": "https://example.com",
            }
        ],
        nli_probs={"entailment": 0.02, "contradiction": 0.95, "neutral": 0.03},
    )

    assert result["verdict"] == "unsupported"


def test_pipeline_report_has_claims_and_evidence():
    pipeline = VerificationPipeline()
    report = pipeline.verify_text("Paris is the capital of France. The Eiffel Tower is in Paris.")
    assert report["document_summary"]["total_claims"] >= 1
    assert report["claims"]
    assert report["claims"][0]["evidence"]


def test_diagnostic_retrieval_does_not_change_scoring_evidence():
    pipeline = VerificationPipeline.__new__(VerificationPipeline)
    candidates = [{"text": f"evidence {index}", "score": 0.9} for index in range(10)]
    pipeline.retriever = Mock()
    pipeline.retriever.search.return_value = candidates
    pipeline._classify_claim = Mock(
        return_value={"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1}
    )
    pipeline.judge = Mock(enabled=False)
    pipeline.risk_scorer = Mock()
    pipeline.risk_scorer.score_claim.return_value = {
        "risk_score": 10,
        "verdict": "supported",
        "why": "supported",
    }

    report = pipeline.verify_text(
        "Paris is the capital of France.",
        diagnostic_retrieval_k=10,
    )

    claim = report["claims"][0]
    assert claim["evidence"] == candidates[:3]
    assert claim["retrieval_candidates"] == candidates
    pipeline.risk_scorer.score_claim.assert_called_once()
    assert pipeline.risk_scorer.score_claim.call_args.args[1] == candidates[:3]


def test_api_health_and_verification():
    client = TestClient(app)
    health = client.get("/health")
    readiness = client.get("/readyz")
    assert health.status_code == 200
    assert readiness.status_code == 200
    payload = {"text": "Paris is the capital of France."}
    response = client.post("/verify", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["document_summary"]["total_claims"] >= 1


def test_nli_model_distinguishes_support_and_contradiction():
    from verimind.data import build_synthetic_dataset

    dataset = build_synthetic_dataset()
    model = NLIModel().fit(dataset["pairs"], dataset["labels"])
    supported, contradicted = model.predict(
        [
            ("Paris is the capital of France.", "Paris is the capital city of France."),
            ("Paris is the capital of France.", "Paris is the capital of Germany."),
        ]
    )
    assert supported == "entailment"
    assert contradicted == "contradiction"


def test_pipeline_maps_probabilities_by_label_for_contradictions():
    pipeline = VerificationPipeline()
    scores = pipeline._classify_claim(
        "Paris is the capital of France.",
        [{"text": "Paris is the capital of Germany."}],
    )
    assert scores["contradiction"] > scores["entailment"]


def test_pipeline_weights_nli_by_relevance_without_mixing_passage_labels():
    pipeline = VerificationPipeline.__new__(VerificationPipeline)
    pipeline.nli_model = Mock(classes_=["entailment", "neutral", "contradiction"])
    pipeline.nli_model.predict_proba.return_value = [
        [0.001, 0.009, 0.990],
        [0.990, 0.009, 0.001],
    ]

    scores = pipeline._classify_claim(
        "CeeLo Green has a song called Fuck You!.",
        [
            {"text": "CeeLo Green's The Good Life is an album by the singer."},
            {"text": "CeeLo Green released the song Fuck You, which became a hit."},
        ],
    )

    assert scores["entailment"] > scores["contradiction"]
    assert scores["contradiction"] < 0.3
    assert sum(scores.values()) == pytest.approx(1.0)


def test_web_search_failure_is_logged_and_local_evidence_is_retained(caplog):
    retriever = HybridRetriever()
    retriever.local.search = Mock(
        return_value=[
            {
                "text": "Paris is the capital of France.",
                "source": "local",
                "domain": "local",
                "score": 0.9,
            }
        ]
    )

    class FailingBackend(SearchBackend):
        def search(self, query: str, k: int = 5) -> list[SearchResult]:
            raise requests.Timeout("search timed out")

    retriever.backend = FailingBackend()

    with caplog.at_level(logging.WARNING, logger="verimind.retrieval"):
        results = retriever.search("Paris is the capital of France.", k=1)

    assert results[0]["source"] == "local"
    assert "continuing with local evidence" in caplog.text


def test_tavily_search_does_not_retry_failed_requests(monkeypatch):
    post = Mock(side_effect=requests.ConnectionError("offline"))
    monkeypatch.setattr("verimind.web_search.requests.post", post)
    backend = TavilySearchBackend(api_key="test-key")

    with pytest.raises(requests.ConnectionError, match="offline"):
        backend.search("sample query")

    post.assert_called_once()


def test_wikipedia_search_returns_clean_evidence_without_an_api_key(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "query": {
            "search": [
                {
                    "pageid": 22989,
                    "title": "Paris",
                    "snippet": "Paris is the <span class=\"searchmatch\">capital</span> city of France.",
                    "timestamp": "2026-11-22T03:57:44Z",
                }
            ]
        }
    }
    get = Mock(return_value=response)
    monkeypatch.setattr("verimind.web_search.requests.get", get)

    results = WikipediaSearchBackend().search("Paris capital France")

    response.raise_for_status.assert_called_once()
    get.assert_called_once()
    assert results == [
        SearchResult(
            text="Paris: Paris is the capital city of France.",
            url="https://en.wikipedia.org/?curid=22989",
            source="https://en.wikipedia.org/?curid=22989",
            domain="wikipedia.org",
            score=0.5,
            published="2026-11-22T03:57:44Z",
        )
    ]


def test_tavily_search_uses_general_web_results_and_api_relevance(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "results": [
            {
                "url": "https://example.org/article",
                "content": "Evidence from a general web source.",
                "score": 0.93,
            }
        ]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("verimind.web_search.requests.post", post)

    results = TavilySearchBackend(api_key="test-key").search("sample claim")

    post.assert_called_once()
    assert results[0].text == "Evidence from a general web source."
    assert results[0].domain == "example.org"
    assert results[0].score == 0.93


def test_tavily_requires_a_key_when_selected(monkeypatch):
    monkeypatch.setenv("SEARCH_BACKEND", "tavily")
    monkeypatch.setenv("SEARCH_API_KEY", "")
    monkeypatch.delenv("WEB_SEARCH_API_KEY", raising=False)

    with pytest.raises(ValueError, match="SEARCH_API_KEY or WEB_SEARCH_API_KEY is required"):
        Settings.from_env()


def test_tavily_backend_does_not_silently_skip_missing_key(monkeypatch):
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("WEB_SEARCH_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SEARCH_API_KEY or WEB_SEARCH_API_KEY is required"):
        TavilySearchBackend().search("sample claim")


def test_tavily_uses_legacy_web_search_key_and_endpoint(monkeypatch):
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "legacy-key")
    monkeypatch.setenv("WEB_SEARCH_API_URL", "https://api.tavily.com/search")

    settings = Settings.from_env()
    backend = TavilySearchBackend(
        api_key=settings.search_api_key,
        endpoint=settings.web_search_api_url,
    )

    assert backend.api_key == "legacy-key"
    assert backend.endpoint == "https://api.tavily.com/search"


def test_production_requires_persistent_database_url(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        Settings.from_env()


def test_production_accepts_configured_database_url(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/verimind")

    settings = Settings.from_env()

    assert settings.database_url == "postgresql://user:pass@localhost/verimind"


def test_api_readiness_reports_database_failure_as_unavailable(monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError

    monkeypatch.setattr("api.main.check_database", Mock(side_effect=SQLAlchemyError("offline")))

    response = TestClient(app).get("/readyz")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database is unavailable"}
