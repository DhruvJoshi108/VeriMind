"""Sequential batch verification using the real API and per-row error handling."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.lib.config import load_config
from dashboard.lib.parsing import ClaimColumnMissingError, read_claims_csv
from dashboard.lib.session import client, connection_sidebar
from dashboard.lib.ui import normalized_verdict

connection_sidebar()

config = load_config()
st.title("Batch check")
st.caption(f"Run up to {config.batch_limit} claims sequentially. Failed rows do not stop the batch.")
uploaded = st.file_uploader("CSV with a claim or text column", type="csv")
pasted = st.text_area("Or paste one claim per line")
claims: list[str] = []
if uploaded is not None:
    try:
        claims = read_claims_csv(uploaded, config.batch_limit)
    except ClaimColumnMissingError as exc:
        st.error(str(exc))
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        st.error("The CSV could not be read. Save it as a standard UTF-8 CSV.")
elif pasted.strip():
    claims = [line.strip() for line in pasted.splitlines() if line.strip()]
if len(claims) > config.batch_limit:
    st.warning(f"Only the first {config.batch_limit} rows will be processed.")
    claims = claims[: config.batch_limit]
oversized = [claim for claim in claims if len(claim) > config.max_characters]
if oversized:
    st.warning(
        f"{len(oversized)} row(s) exceed the {config.max_characters:,}-character API limit "
        "and will be marked as errors without being sent."
    )
st.caption(f"Rows ready: {len(claims)}")

input_signature = tuple(claims)
if st.session_state.get("batch_input_signature") != input_signature:
    st.session_state.pop("batch_results", None)
    st.session_state["batch_input_signature"] = input_signature

if st.button("Run batch", type="primary", disabled=not claims):
    progress = st.progress(0, text="Starting batch…")
    rows = []
    api = client()
    for index, claim in enumerate(claims, start=1):
        if len(claim) > config.max_characters:
            row = {
                "claim": claim,
                "request_id": "",
                "verdict": "error",
                "risk_score": None,
                "error": f"Claim exceeds the {config.max_characters:,}-character API limit.",
            }
        else:
            result = api.verify(claim)
            row = {"claim": claim, "request_id": result.request_id}
            if result.ok and isinstance(result.data, dict):
                claim_results = result.data.get("claims")
                first = next(
                    (item for item in claim_results if isinstance(item, dict)),
                    None,
                ) if isinstance(claim_results, list) else None
                if first is None:
                    row.update(
                        {
                            "verdict": "error",
                            "risk_score": None,
                            "error": "The API response did not contain a claim verdict.",
                        }
                    )
                else:
                    row.update(
                        {
                            "verdict": normalized_verdict(
                                str(first.get("verdict", "unverifiable"))
                            ),
                            "risk_score": first.get("risk_score"),
                            "error": "",
                        }
                    )
            else:
                row.update(
                    {
                        "verdict": "error",
                        "risk_score": None,
                        "error": (
                            result.error.message
                            if result.error
                            else "The API response was invalid."
                        ),
                    }
                )
        rows.append(row)
        progress.progress(index / len(claims), text=f"Checked {index}/{len(claims)}")
    st.session_state["batch_results"] = rows

rows = st.session_state.get("batch_results", [])
if rows:
    results = pd.DataFrame(rows)
    selected = st.multiselect("Filter verdict", sorted(results["verdict"].unique()), default=sorted(results["verdict"].unique()))
    filtered = results[results["verdict"].isin(selected)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(results, x="verdict", title="Verdict distribution"), use_container_width=True)
    st.download_button("Download CSV", results.to_csv(index=False), "verimind-batch.csv", "text/csv")
    st.download_button("Download JSON", json.dumps(rows, indent=2), "verimind-batch.json", "application/json")
