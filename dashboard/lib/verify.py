"""Reusable Verify Claim interface for the dashboard home page."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.lib.config import load_config
from dashboard.lib.parsing import load_evaluations
from dashboard.lib.session import client, connection_sidebar
from dashboard.lib.ui import (
    api_error_message,
    confidence_bar,
    evidence_cards,
    normalized_verdict,
    verdict_badge,
)


@st.cache_data(ttl=30)
def example_claims(directory: str, max_characters: int) -> list[str]:
    values: list[str] = []
    for run in load_evaluations(Path(directory)):
        for record in run.get("records", []):
            claim = record.get("claim_or_answer")
            if isinstance(claim, str) and 0 < len(claim) <= max_characters and claim not in values:
                values.append(claim)
                if len(values) == 10:
                    return values
    return values


def render_verify_page() -> None:
    connection_sidebar()
    config = load_config()
    st.title("Verify claim")
    st.caption("Submit a factual claim to the configured API. Results are not cached.")

    choices = [""] + example_claims(str(config.eval_results_dir), config.max_characters)
    selected = st.selectbox(
        "Example claim from saved evaluation data",
        choices,
        format_func=lambda value: value or "Choose an example",
    )
    if selected and st.session_state.get("verify_claim") != selected:
        st.session_state["verify_claim"] = selected

    with st.form("verify_claim_form"):
        claim = st.text_area(
            "Claim",
            key="verify_claim",
            max_chars=config.max_characters,
            height=160,
            help=f"Maximum {config.max_characters:,} characters, matching the API schema.",
        )
        st.caption(f"{len(claim)}/{config.max_characters} characters")
        submitted = st.form_submit_button("Verify", type="primary")

    if submitted and not claim.strip():
        st.warning("Enter a claim before submitting it for verification.")
    elif submitted:
        with st.spinner("Verifying against the API…"):
            result = client().verify(claim.strip())
        if not result.ok:
            st.session_state["latest_verification"] = {
                "error": result.error,
                "request_id": result.request_id,
            }
        elif not isinstance(result.data, dict):
            st.session_state["latest_verification"] = {
                "message": "The API returned an unexpected response.",
                "request_id": result.request_id,
            }
        else:
            payload = result.data
            claims = payload.get("claims")
            primary = (
                next((item for item in claims if isinstance(item, dict)), None)
                if isinstance(claims, list)
                else None
            )
            if primary is None:
                st.session_state["latest_verification"] = {
                    "message": "The API response did not contain a claim verdict.",
                    "request_id": result.request_id,
                }
            else:
                st.session_state["latest_verification"] = {
                    "claim": claim.strip(),
                    "payload": payload,
                    "primary": primary,
                    "request_id": result.request_id,
                }
                history = st.session_state.setdefault("verification_history", [])
                history.append(
                    {
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "claim": claim.strip(),
                        "verdict": normalized_verdict(str(primary.get("verdict", "unverifiable"))),
                        "risk_score": primary.get("risk_score"),
                        "request_id": result.request_id,
                    }
                )

    latest = st.session_state.get("latest_verification")
    if latest:
        if latest.get("error"):
            api_error_message(latest["error"])
        elif latest.get("message"):
            st.error(latest["message"])
            st.caption(f"Request ID: {latest['request_id']}")
        else:
            payload: dict[str, Any] = latest["payload"]
            primary: dict[str, Any] = latest["primary"]
            degraded = bool(
                payload.get("degraded")
                or payload.get("search_degraded")
                or primary.get("degraded")
                or primary.get("judge_error")
                or primary.get("search_degraded")
            )
            if degraded:
                st.warning("The API reported degraded verification. Treat this result with caution.")
            verdict_badge(str(primary.get("verdict", "unverifiable")))
            confidence_bar(primary)
            st.subheader("Rationale")
            st.write(str(primary.get("why") or primary.get("reason") or "No rationale was returned."))
            summary = payload.get("document_summary")
            latency = summary.get("total_latency_ms") if isinstance(summary, dict) else None
            latency_text = f"{latency} ms" if isinstance(latency, (int, float)) else "not reported"
            st.caption(f"Latency: {latency_text} · Request ID: {latest['request_id']}")
            st.subheader("Evidence")
            judge = primary.get("judge")
            citations = judge.get("citations", []) if isinstance(judge, dict) else []
            evidence = primary.get("evidence", [])
            evidence_cards(
                evidence if isinstance(evidence, list) else [],
                citations if isinstance(citations, list) else [],
            )

    st.subheader("Session history")
    history = st.session_state.get("verification_history", [])
    if history:
        frame = pd.DataFrame(history)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history CSV",
            frame.to_csv(index=False),
            "verimind-history.csv",
            "text/csv",
        )
        if st.button("Clear history"):
            st.session_state["verification_history"] = []
            st.rerun()
    else:
        st.caption("No checks in this browser session yet.")
