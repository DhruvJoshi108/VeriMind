"""VERIMIND dashboard home page; Streamlit discovers the pages directory."""

from __future__ import annotations

import streamlit as st

from dashboard.lib.session import connection_sidebar

st.set_page_config(
    page_title="VERIMIND | Claim verification",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="auto",
)

connection_sidebar()

st.title("Welcome to VERIMIND")
st.markdown(
    """
    <div class="vm-home-hero">
        <p class="vm-home-eyebrow">Evidence-first claim checking</p>
        <p>Explore evidence, review transparent verdicts, and inspect how the system reached its result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.subheader("Choose a workspace")

verify_column, batch_column, evaluation_column = st.columns(3)
with verify_column:
    st.markdown(
        """
        <div class="vm-home-card">
            <h3>Verify a claim</h3>
            <p>Check one claim and review the returned evidence and explanation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Verify_Claim.py", label="Open claim verification")

with batch_column:
    st.markdown(
        """
        <div class="vm-home-card">
            <h3>Batch check</h3>
            <p>Submit a CSV of claims and download the results for review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Batch_Check.py", label="Open batch check")

with evaluation_column:
    st.markdown(
        """
        <div class="vm-home-card">
            <h3>Explore evaluation</h3>
            <p>Review model evaluation results and understand system metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Evaluation.py", label="Open evaluation")

st.write("")
st.subheader("A simple workflow")
step_columns = st.columns(3)
for column, step, description in zip(
    step_columns,
    ("1. Submit", "2. Inspect", "3. Decide"),
    (
        "Enter a concise factual claim or upload a CSV.",
        "Read the verdict, model score, explanation, and evidence passages.",
        "Use the result as research support, not as a substitute for human judgment.",
    ),
    strict=True,
):
    with column:
        st.markdown(
            f"""
            <div class="vm-home-card">
                <h3>{step}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.info(
    "Privacy: claims are sent to the API configured in the sidebar and may be saved by that service. "
    "If its search or judging integrations are enabled, claim text may also be sent to those providers. "
    "Do not submit secrets or sensitive personal information."
)
st.caption(
    "VERIMIND is a decision-support tool. Verdicts and model scores can be wrong; "
    "check the cited evidence before relying on a result."
)
