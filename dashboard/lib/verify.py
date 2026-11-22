"""Reusable Verify Claim interface for the dashboard home page."""

from __future__ import annotations

import html
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.lib.config import load_config
from dashboard.lib.session import client, connection_sidebar
from dashboard.lib.ui import (
    api_error_message,
    evidence_cards,
    model_score_bar,
    normalized_verdict,
    verdict_badge,
    verdict_pill,
)


def _verdict_key(claim: dict[str, Any]) -> str:
    verdict = normalized_verdict(str(claim.get("verdict", "unverifiable")))
    return verdict if verdict in {"supported", "refuted"} else "unverifiable"


def _verdict_label(verdict: str) -> str:
    return {
        "supported": "Supported",
        "refuted": "Refuted",
        "unverifiable": "Insufficient evidence",
    }.get(verdict, "Insufficient evidence")


def _claim_text(claim: dict[str, Any], submitted_text: str, index: int, total: int) -> str:
    value = claim.get("claim")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return submitted_text if total == 1 else f"Claim {index + 1}"


def _risk_score(claim: dict[str, Any]) -> float | None:
    value = claim.get("risk_score")
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(score, 0.0), 100.0) if math.isfinite(score) else None


def _highlighted_claim_text(text: str, claims: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    cursor = 0
    highlighted = False
    for index, claim in enumerate(claims, start=1):
        phrase = claim.get("claim")
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        phrase = phrase.strip()
        start = text.find(phrase, cursor)
        if start < 0:
            continue
        verdict = _verdict_key(claim)
        label = html.escape(f"Claim {index}: {_verdict_label(verdict)}", quote=True)
        chunks.extend(
            (
                html.escape(text[cursor:start]),
                f'<span class="vm-claim-highlight vm-{verdict}" title="{label}">',
                html.escape(phrase),
                f"<sup>{index}</sup></span>",
            )
        )
        cursor = start + len(phrase)
        highlighted = True
    if not highlighted:
        return html.escape(text)
    chunks.append(html.escape(text[cursor:]))
    return "".join(chunks)


def _render_result_overview(
    submitted_text: str,
    payload: dict[str, Any],
    claims: list[dict[str, Any]],
    request_id: str,
) -> None:
    with st.container(border=True):
        st.markdown("#### Text checked")
        st.markdown(
            f'<div class="vm-claim-preview">{_highlighted_claim_text(submitted_text, claims)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Claim highlights reflect the API verdicts. Select a row below to inspect its evidence.")

    counts = Counter(_verdict_key(claim) for claim in claims)
    total = len(claims)
    refuted = counts["refuted"]
    uncertain = counts["unverifiable"]
    if refuted:
        headline = f"{refuted} {'claim is' if refuted == 1 else 'claims are'} refuted by the evidence"
    elif uncertain:
        headline = f"{uncertain} {'claim needs' if uncertain == 1 else 'claims need'} more evidence"
    else:
        headline = f"{total} {'claim is' if total == 1 else 'claims are'} supported by the evidence"

    summary = payload.get("document_summary")
    summary = summary if isinstance(summary, dict) else {}
    risk_value = summary.get("overall_risk_score")
    if isinstance(risk_value, bool):
        overall_risk = None
    else:
        try:
            overall_risk = float(risk_value)
        except (TypeError, ValueError):
            overall_risk = None
    if overall_risk is None or not math.isfinite(overall_risk):
        scores = [score for score in (_risk_score(claim) for claim in claims) if score is not None]
        overall_risk = sum(scores) / len(scores) if scores else None
    if overall_risk is not None:
        overall_risk = min(max(overall_risk, 0.0), 100.0)

    with st.container(border=True):
        st.markdown("#### Result")
        st.markdown(f"### {headline}")
        latency = summary.get("total_latency_ms")
        try:
            latency_seconds = float(latency) / 1000
        except (TypeError, ValueError):
            latency_seconds = None
        timing = f"Checked {total} {'claim' if total == 1 else 'claims'}"
        if latency_seconds is not None and math.isfinite(latency_seconds) and latency_seconds >= 0:
            timing += f" in {latency_seconds:.1f} s"
        st.caption(timing)

        segments = []
        for verdict in ("supported", "refuted", "unverifiable"):
            count = counts[verdict]
            if count:
                width = count / total * 100
                segments.append(
                    f'<span class="vm-{verdict}" style="width:{width:.2f}%" '
                    f'title="{_verdict_label(verdict)}: {count}"></span>'
                )
        st.markdown(f'<div class="vm-verdict-strip">{"".join(segments)}</div>', unsafe_allow_html=True)
        legend = " ".join(
            f"<span><strong>{counts[verdict]}</strong> {_verdict_label(verdict)}</span>"
            for verdict in ("supported", "refuted", "unverifiable")
        )
        st.markdown(f'<div class="vm-verdict-legend">{legend}</div>', unsafe_allow_html=True)

        if overall_risk is not None:
            st.progress(overall_risk / 100, text=f"Overall risk · {overall_risk:.0f}/100")
        st.caption(f"Request ID: {request_id}")


def _render_claim_audit(
    submitted_text: str,
    claims: list[dict[str, Any]],
    request_id: str,
) -> None:
    st.divider()
    st.subheader("Claim-level audit trail")
    table_column, evidence_column = st.columns([1.05, 0.95])
    rows = []
    for index, claim in enumerate(claims):
        score = _risk_score(claim)
        text = _claim_text(claim, submitted_text, index, len(claims))
        rows.append(
            {
                "#": index + 1,
                "Claim": text if len(text) <= 180 else f"{text[:177]}...",
                "Verdict": _verdict_label(_verdict_key(claim)),
                "Risk": f"{score:.0f}/100" if score is not None else "Not reported",
            }
        )

    with table_column:
        st.caption("Select a claim row to inspect its evidence.")
        event = st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=min(420, 42 + 38 * len(rows)),
            on_select="rerun",
            selection_mode="single-row",
            key="verify_claim_table",
        )
        selected_rows = getattr(getattr(event, "selection", None), "rows", [])
        if selected_rows:
            selected_index = selected_rows[0]
            st.session_state["verify_selected_claim_index"] = selected_index
        else:
            selected_index = st.session_state.get("verify_selected_claim_index", 0)
        if not isinstance(selected_index, int) or not 0 <= selected_index < len(claims):
            selected_index = 0
        selected_claim = claims[selected_index]

    with evidence_column:
        text = _claim_text(selected_claim, submitted_text, selected_index, len(claims))
        st.markdown(f"#### Claim {selected_index + 1} evidence")
        st.markdown(verdict_pill(_verdict_key(selected_claim)), unsafe_allow_html=True)
        st.caption("Claim")
        st.markdown(f'<div class="vm-selected-claim">{html.escape(text)}</div>', unsafe_allow_html=True)

        score = _risk_score(selected_claim)
        evidence_value = selected_claim.get("evidence")
        evidence = [item for item in evidence_value if isinstance(item, dict)] if isinstance(evidence_value, list) else []
        metrics = st.columns(2)
        metrics[0].metric("Risk score", f"{score:.0f}/100" if score is not None else "Not reported")
        metrics[1].metric("Evidence passages", len(evidence))
        st.caption("Why this risk score")
        st.write(str(selected_claim.get("why") or selected_claim.get("reason") or "No rationale was returned."))
        verdict_badge(str(selected_claim.get("verdict", "unverifiable")))
        model_score_bar(selected_claim)

        judge = selected_claim.get("judge")
        citations = judge.get("citations", []) if isinstance(judge, dict) else []
        citations = [item for item in citations if isinstance(item, int)] if isinstance(citations, list) else []
        evidence_cards(evidence, citations, expanded=True)
        st.caption(f"Request ID: {request_id}")


def render_verify_page() -> None:
    connection_sidebar()
    config = load_config()
    st.title("Verify Claim")
    st.caption("Paste an answer to split it into claims and check each against retrieved evidence.")
    st.caption(
        "Privacy: claim text is sent to the configured API and may be saved or forwarded to its providers. "
        "Do not submit secrets or sensitive personal information."
    )

    input_column, result_column = st.columns([1.08, 0.92])
    with input_column:
        with st.container(border=True):
            claim = st.text_area(
                "Text to verify",
                key="verify_claim",
                max_chars=config.max_characters,
                height=140,
                placeholder="Paste a factual claim or longer answer to check.",
                help=f"Maximum {config.max_characters:,} characters, matching the API schema.",
            )
            count_column, submit_column = st.columns([0.56, 0.44])
            count_column.caption(f"{len(claim)}/{config.max_characters} characters")
            latest_result = st.session_state.get("latest_verification")
            button_label = (
                "Verify again"
                if latest_result and not latest_result.get("error") and not latest_result.get("message")
                else "Verify claim"
            )
            submitted = submit_column.button(
                button_label,
                type="primary",
                use_container_width=True,
            )

        if submitted and not claim.strip():
            st.warning("Enter a claim before submitting it for verification.")
        elif submitted:
            with st.spinner("Verifying against the API..."):
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
                response_claims = payload.get("claims")
                claims = (
                    [item for item in response_claims if isinstance(item, dict)]
                    if isinstance(response_claims, list)
                    else []
                )
                if not claims:
                    st.session_state["latest_verification"] = {
                        "message": "The API response did not contain a claim verdict.",
                        "request_id": result.request_id,
                    }
                else:
                    st.session_state["latest_verification"] = {
                        "submitted_text": claim.strip(),
                        "payload": payload,
                        "claims": claims,
                        "request_id": result.request_id,
                    }
                    st.session_state["verify_selected_claim_index"] = 0
                    st.session_state.pop("verify_claim_table", None)
                    primary = claims[0]
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

    with result_column:
        latest = st.session_state.get("latest_verification")
        if not latest:
            with st.container(border=True):
                st.markdown("#### Your result will appear here")
                st.write("Submit a claim to see its verdict, overall risk, and supporting evidence.")
                st.caption("Model scores are not calibrated confidence estimates.")
        elif latest.get("error"):
            api_error_message(latest["error"])
        elif latest.get("message"):
            st.error(latest["message"])
            st.caption(f"Request ID: {latest['request_id']}")
        else:
            payload: dict[str, Any] = latest["payload"]
            claims: list[dict[str, Any]] = latest["claims"]
            degraded = bool(
                payload.get("degraded")
                or payload.get("search_degraded")
                or any(
                    item.get("degraded") or item.get("judge_error") or item.get("search_degraded")
                    for item in claims
                )
            )
            if degraded:
                st.warning("The API reported degraded verification. Treat this result with caution.")
            _render_result_overview(
                latest["submitted_text"],
                payload,
                claims,
                latest["request_id"],
            )

    latest = st.session_state.get("latest_verification")
    if latest and not latest.get("error") and not latest.get("message"):
        _render_claim_audit(
            latest["submitted_text"],
            latest["claims"],
            latest["request_id"],
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
