from io import BytesIO
from pathlib import Path

import pytest

from dashboard.lib.config import display_api_origin
from dashboard.lib.parsing import (
    ClaimColumnMissingError,
    load_evaluation_file,
    load_evaluations,
    parse_prometheus,
    read_claims_csv,
)


def test_parse_prometheus_labels_and_values():
    samples = parse_prometheus('metric_total{path="/verify",status="200"} 3\n# ignored\nlatency 1.5\n')

    assert samples[0] == {"name": "metric_total", "labels": {"path": "/verify", "status": "200"}, "value": 3.0}
    assert samples[1]["value"] == 1.5


def test_read_claims_csv_preserves_numeric_text():
    claims = read_claims_csv(BytesIO(b"claim\n00123\n2024\n"), limit=50)

    assert claims == ["00123", "2024"]


def test_read_claims_csv_requires_claim_or_text_column():
    with pytest.raises(ClaimColumnMissingError, match="`claim` or `text`"):
        read_claims_csv(BytesIO(b"statement\nExample claim\n"), limit=50)


def test_load_existing_jsonl_evaluation_artifact():
    root = Path(__file__).resolve().parents[2]
    result = load_evaluation_file(root / "results" / "eval_20261022T100330Z.jsonl")

    assert result["summary"]["accuracy"] == 1 / 3
    assert len(result["records"]) == 60


def test_parse_prometheus_fixture_and_escaped_labels():
    fixture = Path(__file__).parent / "fixtures" / "metrics.prom"
    samples = parse_prometheus(fixture.read_text(encoding="utf-8"))
    assert len(samples) == 3
    assert samples[0]["labels"] == {"method": "POST", "path": "/verify", "status": "200"}


def test_json_evaluation_without_summary_derives_metrics_from_records():
    fixture = Path(__file__).parent / "fixtures" / "evaluation-package.json"
    result = load_evaluation_file(fixture)

    assert result["summary"]["accuracy"] == 2 / 3
    assert result["summary"]["class_metrics"]["refuted"]["recall"] == 0
    assert result["summary"]["retrieval_hit_rate_verifiable_claims"] == 1 / 2
    assert len(result["records"]) == 3


def test_default_evaluation_loader_includes_legacy_and_eval_results():
    root = Path(__file__).resolve().parents[2]
    runs = load_evaluations(root / "eval" / "results")

    assert any(Path(run["path"]).name == "eval_20261022T100330Z.jsonl" for run in runs)


def test_display_api_origin_omits_credentials_path_and_query():
    assert (
        display_api_origin("https://user:secret@example.test:8443/private?token=hidden")
        == "https://example.test:8443"
    )
