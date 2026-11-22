"""Safe parsers for real Prometheus and evaluation artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

_METRIC = re.compile(r"^([\w:]+)(?:\{([^}]*)\})?\s+([-+eE.\d]+)$")
_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
_LABELS = ("supported", "refuted", "unverifiable")


class ClaimColumnMissingError(ValueError):
    """Raised when a batch CSV has no supported claim column."""


def read_claims_csv(source: BinaryIO, limit: int) -> list[str]:
    frame = pd.read_csv(source, nrows=limit + 1, dtype="string")
    column = next((item for item in ("claim", "text") if item in frame.columns), None)
    if column is None:
        raise ClaimColumnMissingError("CSV needs a `claim` or `text` column.")
    return [value.strip() for value in frame[column].dropna() if value.strip()]


def _unescape_label(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except json.JSONDecodeError:
        return value


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = {key: _unescape_label(value) for key, value in _LABEL.findall(raw_labels or "")}
        samples.append({"name": name, "labels": labels, "value": value})
    return samples


def metric_samples(samples: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample["name"] == name]


def load_evaluation_file(path: Path) -> dict[str, Any]:
    """Load existing JSONL or package JSON result formats and validate basic structure."""
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if not all(isinstance(row, dict) for row in rows):
                raise ValueError("JSONL rows must be objects")
            metadata = next((row for row in rows if row.get("record_type") == "run_metadata"), {})
            records = [row for row in rows if row.get("record_type") != "run_metadata"]
        else:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Evaluation JSON must contain an object")
            records = payload.get("records", [])
            metadata = payload.get("metadata", {})
            if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
                raise ValueError("Evaluation records must be a list of objects")
            if not isinstance(metadata, dict):
                raise ValueError("Evaluation metadata must be an object")
            embedded_metadata = next(
                (row for row in records if row.get("record_type") == "run_metadata"), None
            )
            if embedded_metadata is not None:
                metadata = {**embedded_metadata, **metadata}
                records = [row for row in records if row.get("record_type") != "run_metadata"]
            if not records and isinstance(payload.get("results"), list):
                records = payload["results"]
                if not all(isinstance(row, dict) for row in records):
                    raise ValueError("Evaluation results must be a list of objects")
        if not isinstance(metadata, dict):
            raise ValueError("Evaluation metadata must be an object")
        summary = metadata.get("summary", {}) if path.suffix == ".jsonl" else payload.get("summary", {})
        if not isinstance(summary, dict) or not summary:
            summary = metadata.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        if not summary and records:
            summary = summarize_evaluation_records(records)
        return {"path": str(path), "metadata": metadata, "summary": summary, "records": records}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"path": str(path), "error": f"Could not load evaluation file: {exc}", "records": [], "summary": {}}


def summarize_evaluation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive evaluation metrics from persisted individual cases when no summary exists."""
    completed = [
        row
        for row in records
        if row.get("true_label") in _LABELS and row.get("predicted_label") in _LABELS
    ]
    matrix = [[0 for _ in _LABELS] for _ in _LABELS]
    for row in completed:
        matrix[_LABELS.index(row["true_label"])][_LABELS.index(row["predicted_label"])] += 1

    class_metrics: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(_LABELS):
        true_positive = matrix[index][index]
        predicted_count = sum(matrix[row][index] for row in range(len(_LABELS)))
        actual_count = sum(matrix[index])
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_count,
        }

    verifiable = [
        row
        for row in records
        if row.get("true_label") in {"supported", "refuted"}
        and row.get("retrieval_evaluable", row.get("retrieval_hit") is not None)
    ]
    retrieval_hits = sum(row.get("retrieval_hit") is True for row in verifiable)
    return {
        "n_cases": len(records),
        "n_completed": len(completed),
        "n_errors": len(records) - len(completed),
        "accuracy": (
            sum(row["true_label"] == row["predicted_label"] for row in completed) / len(completed)
            if completed
            else None
        ),
        "class_metrics": class_metrics,
        "confusion_matrix_labels": list(_LABELS),
        "confusion_matrix": matrix,
        "retrieval_hit_rate_verifiable_claims": (
            retrieval_hits / len(verifiable) if verifiable else None
        ),
        "retrieval_hit_count": retrieval_hits,
        "retrieval_evaluable_count": len(verifiable),
        "unverifiable_rate": (
            sum(row["predicted_label"] == "unverifiable" for row in completed) / len(completed)
            if completed
            else None
        ),
    }


def load_evaluations(directory: Path) -> list[dict[str, Any]]:
    directories = [directory]
    project_root = Path(__file__).resolve().parents[2]
    if directory.resolve() == project_root / "eval" / "results":
        directories.append(project_root / "results")

    def modified_time(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    files_by_path = {
        path.resolve(): path
        for folder in directories
        if folder.is_dir()
        for path in (*folder.glob("*.json"), *folder.glob("*.jsonl"))
    }
    files = sorted(files_by_path.values(), key=modified_time, reverse=True)
    return [load_evaluation_file(path) for path in files]
