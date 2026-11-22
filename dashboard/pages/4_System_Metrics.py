"""Operational API health and Prometheus metrics page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.lib.api_client import ApiClient, ApiResult
from dashboard.lib.config import load_config
from dashboard.lib.parsing import metric_samples, parse_prometheus
from dashboard.lib.session import connection_sidebar
from dashboard.lib.ui import api_error_message

connection_sidebar()
config = load_config()
st.title("System metrics")
interval = st.selectbox("Auto-refresh", ["Off", "5 seconds", "10 seconds", "30 seconds"])
refresh_seconds = {"Off": None, "5 seconds": 5, "10 seconds": 10, "30 seconds": 30}[interval]


@st.cache_data(ttl=5, show_spinner=False)
def metrics_snapshot(api_url: str, timeout_seconds: float) -> tuple[ApiResult, ApiResult, ApiResult]:
    api = ApiClient(api_url, timeout_seconds=timeout_seconds)
    return api.health(), api.ready(), api.metrics()


def _single_value(samples: list[dict[str, Any]], name: str) -> float | None:
    values = metric_samples(samples, name)
    return values[0]["value"] if values else None


def render_metrics() -> None:
    health, ready, response = metrics_snapshot(
        st.session_state.get("dashboard_api_url", config.api_url),
        config.timeout_seconds,
    )
    left, right = st.columns(2)
    if health.ok:
        left.success("Health: live")
    elif health.error:
        left.error(f"Health: {health.error.message}")
        left.caption(f"Request ID: {health.request_id}")
    if ready.ok:
        right.success("Readiness: ready")
    elif ready.error:
        right.error(f"Readiness: {ready.error.message}")
        right.caption(f"Request ID: {ready.request_id}")
    if not response.ok or not isinstance(response.data, str):
        if response.error:
            api_error_message(response.error)
        else:
            st.error("Metrics response was invalid.")
        return
    samples = parse_prometheus(response.data)
    requests = metric_samples(samples, "verimind_http_requests_total")
    verdicts = metric_samples(samples, "verimind_verdicts_total")
    average = _single_value(samples, "verimind_request_latency_ms")
    flagged = _single_value(samples, "verimind_flagged_claim_ratio")
    errors = _single_value(samples, "verimind_judge_errors_total")
    percentiles = {
        sample["labels"].get("quantile"): sample["value"]
        for sample in metric_samples(samples, "verimind_request_latency_percentile_ms")
    }
    retrieval_latency = _single_value(samples, "verimind_retrieval_latency_ms")
    judge_latency = _single_value(samples, "verimind_judge_latency_ms")
    summary_cols = st.columns(2)
    summary_cols[0].metric(
        "Average request latency",
        f"{average:.1f} ms" if average is not None else "not exported",
    )
    summary_cols[1].metric(
        "Flagged-claim rate",
        f"{flagged:.1%}" if flagged is not None else "not exported",
    )
    percentile_cols = st.columns(3)
    for column, quantile, label in zip(
        percentile_cols,
        ("0.50", "0.95", "0.99"),
        ("p50", "p95", "p99"),
        strict=True,
    ):
        value = percentiles.get(quantile)
        column.metric(f"Request latency {label}", f"{value:.1f} ms" if value is not None else "not exported")
    stage_cols = st.columns(3)
    stage_cols[0].metric("Judge errors", f"{errors:.0f}" if errors is not None else "not exported")
    stage_cols[1].metric(
        "Mean retrieval latency",
        f"{retrieval_latency:.1f} ms" if retrieval_latency is not None else "not reported",
    )
    stage_cols[2].metric(
        "Mean judge latency",
        f"{judge_latency:.1f} ms" if judge_latency is not None else "not reported",
    )
    if requests:
        st.subheader("Requests by route and status")
        st.table(
            pd.DataFrame(
                [{**sample["labels"], "count": sample["value"]} for sample in requests]
            )
        )
    if verdicts:
        st.subheader("Verdicts")
        st.bar_chart(pd.DataFrame([{**sample["labels"], "count": sample["value"]} for sample in verdicts]).set_index("verdict"))
    if not requests and not verdicts:
        st.info("The API has not exported request or verdict samples yet.")

if refresh_seconds and hasattr(st, "fragment"):
    @st.fragment(run_every=refresh_seconds)
    def auto_refresh() -> None:
        render_metrics()
    auto_refresh()
else:
    render_metrics()
