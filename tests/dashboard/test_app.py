from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.lib.api_client import ApiError, ApiResult

ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted((ROOT / "dashboard" / "pages").glob("*.py"))


class FakeClient:
    def __init__(self, *, available: bool = True, verify_error: bool = False):
        self.available = available
        self.verify_error = verify_error

    def health(self):
        return self._availability("live")

    def ready(self):
        return self._availability("ready")

    def metrics(self):
        return (
            ApiResult(data="# TYPE verimind_http_requests_total counter\nverimind_http_requests_total{path=\"/verify\",status=\"200\"} 1\n")
            if self.available
            else ApiResult(error=ApiError("The API is unavailable.", "test-request"), request_id="test-request")
        )

    def verify(self, claim):
        if self.verify_error:
            return ApiResult(
                error=ApiError("The API timed out. Please try again.", "verify-request"),
                request_id="verify-request",
            )
        return ApiResult(
            data={
                "document_summary": {"total_latency_ms": 12.0},
                "claims": [
                    {
                        "claim": claim,
                        "verdict": "supported",
                        "risk_score": 10,
                        "why": "The saved evidence supports the claim.",
                        "nli_probs": {"entailment": 0.9},
                        "evidence": [
                            {
                                "title": "Test source",
                                "text": "Evidence passage.",
                                "source": "https://example.test/source",
                                "score": 0.8,
                            }
                        ],
                        "judge": {"citations": [1]},
                    }
                ],
            },
            request_id="verify-request",
        )

    def _availability(self, _label):
        if self.available:
            return ApiResult(data={"status": "ok"}, request_id="health-request")
        return ApiResult(
            error=ApiError("The API is unavailable. Check the URL and service status.", "down-request"),
            request_id="down-request",
        )


def install_fake_clients(monkeypatch, fake):
    def factory(*_args, **_kwargs):
        return fake

    monkeypatch.setattr("dashboard.lib.session.ApiClient", factory)
    monkeypatch.setattr("dashboard.lib.api_client.ApiClient", factory)


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_each_page_renders_when_api_is_down(page, monkeypatch):
    fake = FakeClient(available=False)
    install_fake_clients(monkeypatch, fake)

    app = AppTest.from_file(str(page), default_timeout=20).run()

    assert not app.exception
    assert any("API unavailable" in element.value for element in app.sidebar.error)


def test_verify_flow_renders_real_shaped_response_and_history(monkeypatch):
    install_fake_clients(monkeypatch, FakeClient())
    app = AppTest.from_file(str(ROOT / "dashboard" / "pages" / "1_Verify_Claim.py")).run()

    verify_button = next(button for button in app.button if button.label == "Verify claim")
    assert not verify_button.disabled
    app.text_area[0].set_value("This is an evidence-backed claim.")
    app.run()
    verify_button = next(button for button in app.button if button.label == "Verify claim")
    assert not verify_button.disabled
    verify_button.click().run()

    assert not app.exception
    assert any(element.value == "Supported" for element in app.success)
    assert any("Cited by the judge" in element.value for element in app.success)
    assert any(element.value == "The saved evidence supports the claim." for element in app.markdown)


def test_verify_page_does_not_preload_sample_claims(monkeypatch):
    install_fake_clients(monkeypatch, FakeClient())
    app = AppTest.from_file(str(ROOT / "dashboard" / "pages" / "1_Verify_Claim.py")).run()

    assert not app.exception
    assert not app.selectbox
    assert any(element.label == "Text to verify" for element in app.text_area)


def test_verify_failure_shows_friendly_error_and_request_id(monkeypatch):
    install_fake_clients(monkeypatch, FakeClient(verify_error=True))
    app = AppTest.from_file(str(ROOT / "dashboard" / "pages" / "1_Verify_Claim.py")).run()

    app.text_area[0].set_value("A factual claim.")
    app.run()
    next(button for button in app.button if button.label == "Verify claim").click().run()

    assert not app.exception
    assert any("API timed out" in element.value for element in app.error)
    assert any("verify-request" in element.value for element in app.caption)


def test_main_entrypoint_shows_home_and_privacy_guidance(monkeypatch):
    install_fake_clients(monkeypatch, FakeClient(available=False))
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=20).run()

    assert not app.exception
    assert any(element.value == "Welcome to VERIMIND" for element in app.title)
    assert any("Choose a workspace" in element.value for element in app.subheader)
    assert any("Do not submit secrets" in element.value for element in app.info)


def test_missing_nli_score_is_not_inferred_from_risk_score(monkeypatch):
    fake = FakeClient()
    fake.verify = lambda _claim: ApiResult(
        data={"claims": [{"verdict": "supported", "risk_score": 10, "why": "Test rationale."}]},
        request_id="verify-request",
    )
    install_fake_clients(monkeypatch, fake)
    app = AppTest.from_file(str(ROOT / "dashboard" / "pages" / "1_Verify_Claim.py")).run()

    app.text_area[0].set_value("A factual claim.")
    app.run()
    next(button for button in app.button if button.label == "Verify claim").click().run()

    assert not app.exception
    assert any("Model score unavailable" in element.value for element in app.caption)


def test_metrics_page_cache_does_not_leak_api_key_into_rendered_text(monkeypatch):
    fake = FakeClient()
    install_fake_clients(monkeypatch, fake)
    st.cache_data.clear()
    app = AppTest.from_file(str(ROOT / "dashboard" / "pages" / "4_System_Metrics.py")).run()

    assert not app.exception
    assert all("secret" not in element.value for element in app.markdown)
