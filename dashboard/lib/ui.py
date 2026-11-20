"""Shared accessible Streamlit components with escaped text-only rendering."""

from __future__ import annotations

import math
from typing import Any

import streamlit as st

from dashboard.lib.api_client import ApiError


def apply_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --vm-navy: #18324b;
            --vm-teal: #16877f;
            --vm-ink: #243447;
            --vm-muted: #687789;
            --vm-border: #e2e8ef;
            --vm-surface: #ffffff;
            --vm-canvas: #f5f7fa;
        }
        .stApp {
            background: var(--vm-canvas);
            color: var(--vm-ink);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stAppViewContainer"] section.main > div {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stAppViewContainer"] .block-container {
            max-width: 1280px;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        h1, h2, h3 {
            color: var(--vm-navy);
            letter-spacing: -0.035em;
        }
        h1 {
            font-size: 2.45rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.3rem !important;
        }
        h2 {
            font-size: 1.55rem !important;
            font-weight: 650 !important;
            margin-top: 1.7rem !important;
        }
        [data-testid="stCaptionContainer"] {
            color: var(--vm-muted);
        }
        div[data-testid="stForm"] {
            background: var(--vm-surface);
            border: 1px solid var(--vm-border);
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 8px 24px rgba(24, 50, 75, 0.045);
        }
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] [role="combobox"] {
            border-radius: 10px;
            border-color: var(--vm-border);
            background: #fbfcfe;
        }
        button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-primaryFormSubmit"] {
            background: var(--vm-teal);
            border: 1px solid var(--vm-teal);
            border-radius: 9px;
            color: #ffffff !important;
            font-weight: 600;
            transition: background 120ms ease, border-color 120ms ease;
        }
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
            background: #116f69;
            border-color: #116f69;
        }
        button[data-testid="stBaseButton-primary"] *,
        button[data-testid="stBaseButton-primaryFormSubmit"] * {
            color: #ffffff !important;
        }
        div[data-testid="stMetric"] {
            background: var(--vm-surface);
            border: 1px solid var(--vm-border);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 5px 16px rgba(24, 50, 75, 0.035);
        }
        [data-testid="stSidebar"] {
            background: var(--vm-navy);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f6f8fb;
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.1rem !important;
            letter-spacing: 0;
            margin-top: 1.15rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #bdc9d5;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            border-bottom: 1px solid rgba(255, 255, 255, 0.17);
            padding-bottom: 1rem;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            border-radius: 8px;
            color: #e8eef5 !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a * {
            color: #e8eef5 !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: rgba(22, 135, 127, 0.35);
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] ul > li:first-child {
            display: none;
        }
        [data-testid="stSidebar"] div[data-testid="stTextInput"] input {
            color: var(--vm-ink);
            background: #ffffff;
        }
        [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
            color: var(--vm-navy);
            background: #ffffff;
            border-color: #d7e0e8;
            border-radius: 8px;
        }
        [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] * {
            color: var(--vm-navy) !important;
        }
        .vm-brand {
            padding: 1.2rem 0 1.1rem;
        }
        .vm-brand-name {
            color: #ffffff;
            font-size: 1.12rem;
            font-weight: 750;
            letter-spacing: 0.13em;
        }
        .vm-brand-caption {
            color: #bdc9d5;
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }
        @media (max-width: 800px) {
            [data-testid="stAppViewContainer"] .block-container {
                padding-left: 1.25rem;
                padding-right: 1.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalized_verdict(value: str) -> str:
    return {"unsupported": "refuted", "uncertain": "unverifiable"}.get(value.lower(), value.lower())


def verdict_badge(verdict: str, degraded: bool = False) -> None:
    label = normalized_verdict(verdict)
    if degraded or label == "degraded":
        st.warning("Degraded")
    elif label == "error":
        st.warning("Error")
    elif label == "supported":
        st.success("Supported")
    elif label == "refuted":
        st.error("Refuted")
    else:
        st.info("Unverifiable")


def api_error_message(error: ApiError) -> None:
    st.error(error.message)
    if error.retry_after:
        st.info(f"Retry after: {error.retry_after}")
    st.caption(f"Request ID: {error.request_id}")


def confidence_bar(claim: dict[str, Any]) -> None:
    probabilities = claim.get("nli_probs", {})
    verdict = normalized_verdict(str(claim.get("verdict", "unverifiable")))
    key = {"supported": "entailment", "refuted": "contradiction", "unverifiable": "neutral"}.get(verdict, "neutral")
    try:
        confidence = float(
            probabilities.get(key, max(0, 1 - float(claim.get("risk_score", 100)) / 100))
        )
    except (AttributeError, TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    st.progress(confidence, text=f"Confidence: {confidence:.0%}")


def evidence_cards(evidence: list[dict[str, Any]], citations: list[int] | None = None) -> None:
    if not evidence:
        st.caption("No evidence passages were returned.")
        return
    cited_indices = set(citations or [])
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("doc_id") or f"Evidence {index}")
        page = item.get("page", item.get("passage_id"))
        if page is not None:
            title = f"{title} · page {page}"
        with st.expander(title):
            st.write(str(item.get("text", "No passage text.")))
            if index in cited_indices:
                st.success("Cited by the judge")
            source = item.get("source")
            if isinstance(source, str) and source.startswith(("https://", "http://")):
                st.link_button("Open source", source)
            try:
                score = float(item.get("rrf_score", item.get("score", 0)))
            except (TypeError, ValueError):
                score = 0.0
            st.caption(f"Retrieval score: {score:.3f}")
            sources = item.get("retrieval_sources")
            if sources:
                st.caption(f"Retrieved by: {sources}")
