from fastapi.testclient import TestClient

from api.deps import get_pipeline
from api.main import create_app


class FakePipeline:
    def verify_text(self, _text: str) -> dict:
        return {
            "document_summary": {
                "total_claims": 1,
                "flagged_claims": 0,
                "overall_risk_score": 10.0,
                "total_latency_ms": 1.0,
            },
            "claims": [
                {
                    "claim": "A claim.",
                    "verdict": "supported",
                    "risk_score": 10.0,
                    "evidence": [],
                }
            ],
        }


def test_app_factory_is_lazy_and_exposes_operational_endpoints(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    assert app.state.pipeline is None
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json() == {"status": "ready"}
    response = client.post("/verify", json={"text": "A factual claim."})

    assert response.status_code == 200
    assert response.json()["report_id"] == 1
    assert response.headers["x-request-id"]
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "verimind_http_requests_total" in metrics.text
    assert 'verimind_verdicts_total{verdict="supported"} 1' in metrics.text


def test_root_serves_the_claim_verification_ui():
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Check a claim against evidence." in response.text
    assert 'id="verify-form"' in response.text
