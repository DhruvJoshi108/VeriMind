"""Shared accessible Streamlit components with escaped text-only rendering."""

from __future__ import annotations

import html
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
            padding-top: clamp(1.25rem, 2.5vw, 2rem);
            padding-bottom: 3rem;
        }
        [data-testid="stAppViewContainer"] .block-container {
            max-width: 1200px;
            padding-left: clamp(1rem, 4vw, 3rem);
            padding-right: clamp(1rem, 4vw, 3rem);
        }
        h1, h2, h3 {
            color: var(--vm-navy);
            letter-spacing: -0.035em;
        }
        h1 {
            font-size: clamp(1.8rem, 3.5vw, 2.45rem) !important;
            font-weight: 700 !important;
            margin-bottom: 0.3rem !important;
        }
        h2 {
            font-size: clamp(1.25rem, 2.5vw, 1.55rem) !important;
            font-weight: 650 !important;
            margin-top: 1.7rem !important;
        }
        h3 {
            font-size: clamp(1.05rem, 2vw, 1.25rem) !important;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stAlert"],
        [data-testid="stCaptionContainer"] {
            min-width: 0;
            overflow-wrap: anywhere;
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
            height: 100%;
            min-width: 0;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            min-width: 0;
            overflow-wrap: anywhere;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stPlotlyChart"] {
            max-width: 100%;
            min-width: 0;
        }
        [data-testid="stButton"],
        [data-testid="stDownloadButton"],
        [data-testid="stFormSubmitButton"] {
            max-width: 100%;
        }
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button,
        [data-testid="stFormSubmitButton"] button {
            white-space: normal;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--vm-border);
        }
        [data-testid="stSidebar"] {
            width: 14.5rem !important;
            min-width: 14.5rem !important;
            max-width: 14.5rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--vm-ink);
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.1rem !important;
            letter-spacing: 0;
            margin-top: 1.15rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--vm-muted);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            border-bottom: 1px solid var(--vm-border);
            padding-bottom: 1rem;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            border-radius: 8px;
            color: var(--vm-ink) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a * {
            color: var(--vm-ink) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background: #f1f5f4;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #e7f3f1;
            color: var(--vm-teal) !important;
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
            color: var(--vm-navy);
            font-size: 1.12rem;
            font-weight: 750;
            letter-spacing: 0.13em;
        }
        .vm-brand-caption {
            color: var(--vm-muted);
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }
        .vm-home-hero {
            padding: clamp(1.5rem, 4vw, 3rem);
            border: 1px solid var(--vm-border);
            border-radius: 22px;
            background: linear-gradient(120deg, #ffffff 0%, #f0f8f6 100%);
            box-shadow: 0 12px 32px rgba(24, 50, 75, 0.06);
        }
        .vm-home-eyebrow {
            margin: 0 0 0.6rem;
            color: var(--vm-teal);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .vm-home-hero p:last-child {
            max-width: 720px;
            margin: 0;
            color: var(--vm-muted);
            font-size: 1.05rem;
        }
        .vm-home-card {
            height: 100%;
            padding: 1.25rem;
            border: 1px solid var(--vm-border);
            border-radius: 14px;
            background: var(--vm-surface);
        }
        .vm-home-card p {
            min-height: 3.2em;
            color: var(--vm-muted);
        }
        .vm-result-card {
            min-height: 100%;
            padding: 1.25rem;
            border: 1px solid var(--vm-border);
            border-radius: 14px;
            background: var(--vm-surface);
            box-shadow: 0 5px 16px rgba(24, 50, 75, 0.035);
        }
        .vm-claim-preview {
            min-height: 8rem;
            padding: 1rem;
            border: 1px solid var(--vm-border);
            border-radius: 10px;
            background: #fbfcfe;
            color: var(--vm-ink);
            line-height: 2;
            overflow-wrap: anywhere;
            white-space: pre-wrap;
        }
        .vm-claim-highlight {
            padding: 0.12rem 0.22rem;
            border-radius: 4px;
            box-decoration-break: clone;
            -webkit-box-decoration-break: clone;
        }
        .vm-claim-highlight.vm-supported {
            background: #e4f4e8;
            box-shadow: inset 0 -2px #38a45a;
        }
        .vm-claim-highlight.vm-refuted {
            background: #fde8e7;
            box-shadow: inset 0 -2px #d94841;
        }
        .vm-claim-highlight.vm-unverifiable {
            background: #fff3d4;
            box-shadow: inset 0 -2px #e6a700;
        }
        .vm-claim-highlight sup {
            margin-left: 0.2rem;
            color: #526170;
            font-size: 0.68em;
            font-weight: 700;
        }
        .vm-verdict-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            line-height: 1.3;
            white-space: nowrap;
        }
        .vm-verdict-pill::before {
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 50%;
            background: currentColor;
            content: "";
        }
        .vm-verdict-pill.vm-supported {
            background: #e4f4e8;
            color: #267c3d;
        }
        .vm-verdict-pill.vm-refuted {
            background: #fde8e7;
            color: #b5332e;
        }
        .vm-verdict-pill.vm-unverifiable {
            background: #fff3d4;
            color: #916400;
        }
        .vm-verdict-strip {
            display: flex;
            overflow: hidden;
            height: 0.65rem;
            margin: 0.8rem 0;
            border-radius: 999px;
            background: #e9edf0;
        }
        .vm-verdict-strip span { min-width: 0; }
        .vm-verdict-strip .vm-supported { background: #2f9e44; }
        .vm-verdict-strip .vm-refuted { background: #d94841; }
        .vm-verdict-strip .vm-unverifiable { background: #f0ad24; }
        .vm-verdict-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem 1rem;
            color: var(--vm-muted);
            font-size: 0.78rem;
        }
        .vm-verdict-legend strong { color: var(--vm-ink); }
        .vm-claim-row {
            padding: 0.2rem 0;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }
        .vm-selected-claim {
            padding: 0.9rem 1rem;
            border-left: 3px solid var(--vm-teal);
            border-radius: 0 8px 8px 0;
            background: #f0f8f6;
            color: var(--vm-ink);
            overflow-wrap: anywhere;
        }
        @media (max-width: 1100px) {
            [data-testid="stAppViewContainer"] .block-container {
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.75rem;
            }
            [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 calc(50% - 0.75rem) !important;
                width: auto !important;
                max-width: 100% !important;
                min-width: 0 !important;
            }
        }
        @media (max-width: 600px) {
            [data-testid="stAppViewContainer"] .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex-basis: 100% !important;
                width: 100% !important;
            }
            div[data-testid="stForm"] {
                padding: 1rem;
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


def verdict_pill(verdict: str) -> str:
    label = normalized_verdict(verdict)
    if label == "supported":
        display, style = "Supported", "supported"
    elif label == "refuted":
        display, style = "Refuted", "refuted"
    else:
        display, style = "Insufficient evidence", "unverifiable"
    return f'<span class="vm-verdict-pill vm-{style}">{html.escape(display)}</span>'


def api_error_message(error: ApiError) -> None:
    st.error(error.message)
    if error.retry_after:
        st.info(f"Retry after: {error.retry_after}")
    st.caption(f"Request ID: {error.request_id}")


def model_score_bar(claim: dict[str, Any]) -> None:
    probabilities = claim.get("nli_probs", {})
    verdict = normalized_verdict(str(claim.get("verdict", "unverifiable")))
    key = {"supported": "entailment", "refuted": "contradiction", "unverifiable": "neutral"}.get(verdict, "neutral")
    if not isinstance(probabilities, dict) or key not in probabilities:
        st.caption("Model score unavailable; this is not a calibrated confidence estimate.")
        return
    try:
        confidence = float(probabilities[key])
    except (AttributeError, TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    st.progress(confidence, text=f"Selected-class model score (not calibrated): {confidence:.0%}")


def evidence_cards(
    evidence: list[dict[str, Any]],
    citations: list[int] | None = None,
    *,
    expanded: bool = False,
) -> None:
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
        with st.expander(title, expanded=expanded):
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
