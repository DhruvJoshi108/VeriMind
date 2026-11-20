from verimind.metrics import Metrics


def test_metrics_export_percentiles_and_real_stage_aggregates():
    metrics = Metrics()
    for latency in (1.0, 2.0, 3.0, 4.0):
        metrics.observe_request("POST", "/verify", 200, latency)
    metrics.observe_report(
        {
            "claims": [
                {
                    "verdict": "supported",
                    "risk_score": 10,
                    "retrieval_latency_ms": 4.0,
                    "judge_latency_ms": 8.0,
                    "judge_error": True,
                },
                {
                    "verdict": "refuted",
                    "risk_score": 90,
                    "retrieval_latency_ms": 6.0,
                    "judge_latency_ms": 12.0,
                    "judge_error": False,
                },
            ]
        }
    )

    rendered = metrics.render_prometheus()

    assert 'verimind_request_latency_percentile_ms{quantile="0.50"} 2.500' in rendered
    assert 'verimind_request_latency_percentile_ms{quantile="0.95"} 3.850' in rendered
    assert "verimind_retrieval_latency_ms 5.000" in rendered
    assert "verimind_judge_latency_ms 10.000" in rendered
    assert "verimind_judge_errors_total 1" in rendered
    assert 'verimind_verdicts_total{verdict="refuted"} 1' in rendered
