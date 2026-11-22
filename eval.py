"""Evaluate the offline VERIMIND pipeline against a held-out FEVER sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from verimind.config import settings
from verimind.pipeline import VerificationPipeline

ROOT = Path(__file__).resolve().parent
EVAL_CASES = ROOT / "data" / "eval_cases.jsonl"
RESULTS_DIR = ROOT / "results"
SEED = 42
LABELS = ("supported", "refuted", "unverifiable")
RETRIEVAL_K_VALUES = (1, 3, 5, 10)


def read_cases(split: str, limit: int | None) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in EVAL_CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [case for case in cases if case["split"] == split]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No evaluation cases available for split {split!r}")
    return selected


def normalize_wiki_title(value: str) -> str:
    return unquote(value).rsplit("/", 1)[-1].replace("_", " ").strip().casefold()


def retrieval_hit(passages: list[dict[str, Any]], gold_pages: list[str]) -> bool | None:
    if not gold_pages:
        return None
    gold_titles = {normalize_wiki_title(page) for page in gold_pages}
    return any(
        normalize_wiki_title(str(passage.get("doc_id", ""))) in gold_titles
        or normalize_wiki_title(str(passage.get("title", ""))) in gold_titles
        or normalize_wiki_title(str(passage.get("source", ""))) in gold_titles
        for passage in passages
    )


def local_verdict(verdict: str) -> str:
    return {
        "supported": "supported",
        "unsupported": "refuted",
        "uncertain": "unverifiable",
    }[verdict]


def evaluation_mode(*, baseline: bool, search_backend: str, judge_backend: str) -> str:
    if baseline:
        return "no_evidence_local_control"
    if search_backend != "null" or judge_backend != "local":
        return "configured_provider_pipeline"
    return "offline_verimind_pipeline"


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials == 0:
        return None
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = (
        z
        * (proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) ** 0.5
        / denominator
    )
    return [center - margin, center + margin]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["predicted_label"] in LABELS]
    errors = [record for record in records if record["error"] is not None]
    y_true = [record["true_label"] for record in completed]
    y_pred = [record["predicted_label"] for record in completed]
    accuracy = float(accuracy_score(y_true, y_pred)) if completed else None
    if completed:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, zero_division=0
        )
        matrix = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    else:
        precision = recall = f1 = [0.0] * len(LABELS)
        support = [0] * len(LABELS)
        matrix = [[0] * len(LABELS) for _ in LABELS]
    class_metrics = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }

    refuted = [record for record in completed if record["true_label"] == "refuted"]
    dangerous_errors = sum(record["predicted_label"] == "supported" for record in refuted)
    verifiable = [
        record
        for record in records
        if record["true_label"] in {"supported", "refuted"}
        and record.get("retrieval_evaluable", record.get("retrieval_hit") is not None)
    ]
    retrieval_hits = [record for record in verifiable if record["retrieval_hit"]]
    retrieval_misses = [record for record in verifiable if not record["retrieval_hit"]]
    verifiable_without_gold = [
        record
        for record in records
        if record["true_label"] in {"supported", "refuted"}
        and not record.get("retrieval_evaluable", record.get("retrieval_hit") is not None)
    ]

    def accuracy_for(group: list[dict[str, Any]]) -> dict[str, Any]:
        judged = [record for record in group if record["predicted_label"] in LABELS]
        correct = sum(record["true_label"] == record["predicted_label"] for record in judged)
        return {
            "accuracy": correct / len(judged) if judged else None,
            "n": len(judged),
            "errors": len(group) - len(judged),
        }

    unverifiable_count = sum(record["predicted_label"] == "unverifiable" for record in completed)
    unverifiable_rate = unverifiable_count / len(completed) if completed else None
    retrieval_recall_at_k = {}
    for k in RETRIEVAL_K_VALUES:
        key = str(k)
        measured = [
            record["retrieval_hits_at_k"][key]
            for record in verifiable
            if key in record.get("retrieval_hits_at_k", {})
            and record["retrieval_hits_at_k"][key] is not None
        ]
        retrieval_recall_at_k[key] = (
            sum(measured) / len(measured) if measured else None
        )
    true_unverifiable_rate = (
        sum(record["true_label"] == "unverifiable" for record in records) / len(records)
    )
    return {
        "n_cases": len(records),
        "n_completed": len(completed),
        "n_errors": len(errors),
        "error_categories": {
            category: sum(record["error"]["category"] == category for record in errors)
            for category in sorted({record["error"]["category"] for record in errors})
        },
        "accuracy": accuracy,
        "accuracy_wilson_95_ci": (
            wilson_interval(
                sum(record["true_label"] == record["predicted_label"] for record in completed),
                len(completed),
            )
            if completed
            else None
        ),
        "class_metrics": class_metrics,
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": matrix,
        "unverifiable_rate": unverifiable_rate,
        "unverifiable_rate_flagged": (
            unverifiable_rate is not None and unverifiable_rate > true_unverifiable_rate
        ),
        "true_unverifiable_rate": true_unverifiable_rate,
        "dangerous_error_rate_refuted_predicted_supported": (
            dangerous_errors / len(refuted) if refuted else None
        ),
        "dangerous_errors": dangerous_errors,
        "refuted_cases": len(refuted),
        "retrieval_hit_rate_verifiable_claims": (
            len(retrieval_hits) / len(verifiable) if verifiable else None
        ),
        "retrieval_hit_count": len(retrieval_hits),
        "retrieval_evaluable_count": len(verifiable),
        "retrieval_recall_at_k": retrieval_recall_at_k,
        "retrieval_cases_without_gold_evidence": len(verifiable_without_gold),
        "judge_accuracy_when_retrieval_hit": accuracy_for(retrieval_hits),
        "judge_accuracy_when_retrieval_missed": accuracy_for(retrieval_misses),
        "mean_latency_seconds": (
            sum(record["latency_seconds"] for record in records) / len(records)
        ),
    }


def save_confusion_matrix(summary: dict[str, Any]) -> None:
    matrix = summary["confusion_matrix"]
    labels = summary["confusion_matrix_labels"]
    cell_size = 150
    left, top = 170, 100
    image = Image.new(
        "RGB",
        (left + cell_size * len(labels) + 30, top + cell_size * len(labels) + 90),
        "white",
    )
    draw = ImageDraw.Draw(image)
    draw.text((left, 25), "VERIMIND FEVER evaluation", fill="black")
    draw.text((left + 55, top - 30), "Predicted label", fill="black")
    draw.text((15, top - 35), "True label", fill="black")
    max_value = max((max(row) for row in matrix), default=0)
    for row, label in enumerate(labels):
        y = top + row * cell_size
        draw.text((15, y + cell_size // 2), label, fill="black")
        for column, predicted in enumerate(labels):
            x = left + column * cell_size
            value = matrix[row][column]
            intensity = int(245 - 150 * value / max_value) if max_value else 245
            draw.rectangle(
                (x, y, x + cell_size, y + cell_size),
                fill=(intensity, intensity + 5, 255),
                outline="white",
                width=3,
            )
            draw.text((x + cell_size // 2, y + cell_size // 2), str(value), fill="black", anchor="mm")
            if row == 0:
                draw.text((x + 10, top - 18), predicted, fill="black")
    image.save(RESULTS_DIR / "confusion_matrix.png")


def evaluate_case(
    pipeline: VerificationPipeline,
    case: dict[str, Any],
    *,
    baseline: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "id": case["id"],
        "claim_or_answer": case["claim_or_answer"],
        "question": case["question"],
        "true_label": case["true_label"],
        "source_note": case["source_note"],
        "gold_evidence": case["gold_evidence"],
        "predicted_label": None,
        "retrieved_passages": [],
        "retrieval_hit": False,
        "retrieval_evaluable": (
            case["true_label"] in {"supported", "refuted"}
            and bool(case["gold_evidence"])
        ),
        "judge_reasoning": None,
        "latency_seconds": None,
        "error": None,
    }
    try:
        report = pipeline.verify_text(
            case["claim_or_answer"],
            diagnostic_retrieval_k=max(RETRIEVAL_K_VALUES),
        )
        passages = [
            passage
            for claim in report["claims"]
            for passage in claim.get("retrieval_candidates", claim["evidence"])
        ]
        record["retrieved_passages"] = passages
        record["retrieval_hits_at_k"] = {
            str(k): retrieval_hit(passages[:k], case["gold_evidence"])
            for k in RETRIEVAL_K_VALUES
        }
        record["retrieval_hit"] = record["retrieval_hits_at_k"]["3"]
        if baseline:
            probabilities = pipeline.nli_model.predict_proba([(case["claim_or_answer"], "")])[0]
            scores = {
                label: float(score)
                for label, score in zip(pipeline.nli_model.classes_, probabilities, strict=True)
            }
            nli_label = max(scores, key=scores.get)
            record["predicted_label"] = {
                "entailment": "supported",
                "contradiction": "refuted",
                "neutral": "unverifiable",
            }[nli_label]
            record["judge_reasoning"] = {
                "mode": "local NLI model without evidence (not an LLM baseline)",
                "nli_probabilities": scores,
            }
        else:
            claim_results = report["claims"]
            if not claim_results:
                record["error"] = {
                    "category": "invalid_pipeline_output",
                    "message": "The pipeline returned no claim verdict.",
                }
            else:
                predicted = [local_verdict(claim["verdict"]) for claim in claim_results]
                if len(set(predicted)) == 1:
                    record["predicted_label"] = predicted[0]
                else:
                    record["error"] = {
                        "category": "multi_label_output",
                        "message": "The pipeline returned multiple distinct claim verdicts.",
                        "claim_verdicts": predicted,
                    }
                record["judge_reasoning"] = [
                    {
                        "claim": claim["claim"],
                        "verdict": claim["verdict"],
                        "risk_score": claim["risk_score"],
                        "why": claim["why"],
                        "nli_probabilities": claim["nli_probs"],
                    }
                    for claim in claim_results
                ]
    # Record unexpected per-case failures instead of dropping cases from the run.
    except Exception as exc:  # noqa: BLE001
        record["error"] = {
            "category": "pipeline_error",
            "type": type(exc).__name__,
            "message": str(exc),
        }
    record["latency_seconds"] = time.perf_counter() - started
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, help="Evaluate only the first N cases in the selected split.")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run a local no-evidence NLI control; this is not a plain-LLM baseline.",
    )
    args = parser.parse_args()
    if args.n is not None and args.n < 1:
        parser.error("--n must be positive")

    cases = read_cases(args.split, args.n)
    pipeline = VerificationPipeline()
    records = [
        evaluate_case(pipeline, case, baseline=args.baseline)
        for case in cases
    ]
    summary = summarize(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"eval_{timestamp}.jsonl"
    metadata = {
        "record_type": "run_metadata",
        "timestamp_utc": timestamp,
        "split": args.split,
        "seed": SEED,
        "model": (
            settings.nli_model_name
            if settings.nli_backend == "transformer"
            else "scikit-learn MLPClassifier (local model artifact)"
        ),
        "mode": evaluation_mode(
            baseline=args.baseline,
            search_backend=settings.search_backend,
            judge_backend=settings.judge_backend,
        ),
        "baseline_is_plain_llm": False,
        "retriever": (
            f"local HybridRetriever with {settings.search_backend} search"
            if settings.search_backend != "null"
            else "local HybridRetriever only"
        ),
        "configuration": {
            "seed": SEED,
            "nli_backend": settings.nli_backend,
            "nli_model_path": settings.nli_model_path,
            "search_backend": settings.search_backend,
            "judge_backend": settings.judge_backend,
            "nli_random_state": getattr(pipeline.nli_model, "random_state", None),
            "training_data": (
                "MultiNLI-pretrained DeBERTa checkpoint"
                if settings.nli_backend == "transformer"
                else (
                    "selected NLI_MODEL_PATH artifact; see its training run for provenance"
                    if settings.nli_model_path
                    else "verimind.data.build_synthetic_dataset"
                )
            ),
            "retrieval_evidence_limit_per_claim": 3,
            "retrieval_diagnostic_limit": max(RETRIEVAL_K_VALUES),
            "risk_flag_threshold": 60,
        },
        "temperature": None,
        "summary": summary,
    }
    with output_path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    save_confusion_matrix(summary)

    print(json.dumps({"result_file": str(output_path), **metadata}, indent=2, ensure_ascii=False))
    return 1 if summary["n_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
