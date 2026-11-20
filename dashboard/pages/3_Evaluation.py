"""Render only persisted, real evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.lib.config import load_config
from dashboard.lib.parsing import load_evaluations
from dashboard.lib.session import connection_sidebar

connection_sidebar()
config = load_config()
st.title("Evaluation")
st.caption(f"Evaluation source: {config.eval_results_dir}")


@st.cache_data(ttl=30)
def evaluations(path: str) -> list[dict[str, Any]]:
    return load_evaluations(Path(path))


runs = evaluations(str(config.eval_results_dir))
if not runs:
    st.warning(
        "No evaluation files were found. Run `python eval.py --split test` to create a real result artifact."
    )
    st.stop()
valid = [run for run in runs if not run.get("error")]
invalid_count = len(runs) - len(valid)
if invalid_count:
    st.warning(f"{invalid_count} evaluation file(s) could not be read.")
if not valid:
    st.error(runs[0].get("error", "No readable evaluation artifact."))
    st.stop()

choice = st.selectbox(
    "Evaluation run",
    range(len(valid)),
    format_func=lambda index: valid[index]["path"],
)
run = valid[choice]
summary = run.get("summary", {})
records = run.get("records", [])
if not summary:
    st.warning("The selected artifact contains no evaluable summary or case records.")
    st.stop()

classes = summary.get("class_metrics", {})
macro_f1 = (
    sum(float(item["f1"]) for item in classes.values() if isinstance(item, dict) and item.get("f1") is not None)
    / len(classes)
    if classes
    else None
)
accuracy = summary.get("accuracy")
recall = summary.get("retrieval_hit_rate_verifiable_claims")
cols = st.columns(4)
cols[0].metric("Accuracy", f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else "not reported")
cols[1].metric("Macro F1", f"{macro_f1:.3f}" if macro_f1 is not None else "not reported")
cols[2].metric("Gold-page recall@k", f"{recall:.1%}" if isinstance(recall, (int, float)) else "not reported")
cols[3].metric("Completed cases", summary.get("n_completed", "not reported"))

matrix = summary.get("confusion_matrix")
matrix_labels = summary.get("confusion_matrix_labels")
if isinstance(matrix, list) and matrix and isinstance(matrix_labels, list) and matrix_labels:
    st.subheader("Confusion matrix")
    st.plotly_chart(
        px.imshow(
            matrix,
            x=matrix_labels,
            y=matrix_labels,
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Cases"},
        ),
        use_container_width=True,
    )
if isinstance(classes, dict) and classes:
    st.subheader("Per-class precision and recall")
    class_frame = pd.DataFrame(
        [{"label": label, **values} for label, values in classes.items() if isinstance(values, dict)]
    )
    st.dataframe(class_frame, use_container_width=True, hide_index=True)

st.subheader("Before / after baseline comparison")
baseline_modes = {"no_evidence_local_control", "local_no_evidence_baseline"}
baselines = [
    item
    for item in valid
    if item.get("metadata", {}).get("mode") in baseline_modes
]
selected_mode = run.get("metadata", {}).get("mode")
if selected_mode in baseline_modes:
    full_runs = [item for item in valid if item.get("metadata", {}).get("mode") not in baseline_modes]
    comparison_run = full_runs[0] if full_runs else None
    baseline_run = run
else:
    comparison_run = run
    baseline_run = next(
        (
            item
            for item in baselines
            if item.get("metadata", {}).get("split") == run.get("metadata", {}).get("split")
        ),
        None,
    )

if comparison_run and baseline_run:
    comparisons = []
    for label, item in (("Baseline: local no-evidence control", baseline_run), ("Full pipeline", comparison_run)):
        item_summary = item.get("summary", {})
        comparisons.append(
            {
                "run": label,
                "accuracy": item_summary.get("accuracy"),
                "unverifiable prediction rate": item_summary.get("unverifiable_rate"),
                "gold-page recall@k": item_summary.get("retrieval_hit_rate_verifiable_claims"),
                "cases": item_summary.get("n_completed"),
                "artifact": Path(item["path"]).name,
            }
        )
    st.dataframe(pd.DataFrame(comparisons), use_container_width=True, hide_index=True)
    st.caption(
        "The saved baseline is a local no-evidence NLI control, not a same-model LLM baseline."
    )
else:
    st.info("A matching saved no-evidence baseline run was not found; no comparison values are inferred.")

st.subheader("Same-model LLM comparison")
llm_baseline_modes = {"same_model_llm_no_retrieval", "same_model_llm_baseline"}
llm_full_modes = {"same_model_llm_full_pipeline", "same_model_llm_with_retrieval"}
llm_baselines = [item for item in valid if item.get("metadata", {}).get("mode") in llm_baseline_modes]
llm_full = [item for item in valid if item.get("metadata", {}).get("mode") in llm_full_modes]
if llm_baselines and llm_full:
    st.dataframe(
        pd.DataFrame(
            [
                {"run": "Same-model LLM without retrieval", **llm_baselines[0].get("summary", {})},
                {"run": "Same-model LLM with retrieval", **llm_full[0].get("summary", {})},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "Not run yet: this repository's evaluator does not implement a same-model LLM comparison. "
        "The available `python eval.py --split test --baseline` command runs a local NLI control "
        "and must not be interpreted as that LLM comparison."
    )

st.subheader("Individual predictions")
if records:
    frame = pd.DataFrame(records)
    only_errors = st.checkbox("Show wrong verdicts, errors, or missed gold pages")
    if only_errors:
        wrong = frame["true_label"] != frame["predicted_label"] if {"true_label", "predicted_label"} <= set(frame.columns) else pd.Series(False, index=frame.index)
        errors = frame["error"].notna() & (frame["error"] != "") if "error" in frame else pd.Series(False, index=frame.index)
        missed = (
            frame["retrieval_hit"].eq(False)
            if "retrieval_hit" in frame
            else pd.Series(False, index=frame.index)
        )
        frame = frame[wrong | errors | missed]
    columns = [
        column
        for column in (
            "id",
            "claim_or_answer",
            "true_label",
            "predicted_label",
            "retrieval_hit",
            "error",
        )
        if column in frame
    ]
    st.dataframe(frame[columns], use_container_width=True, hide_index=True)
else:
    st.caption("The selected evaluation artifact contains no individual predictions.")
