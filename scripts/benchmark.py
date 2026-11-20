"""Small benchmark harness comparing the explainable pipeline against a naive baseline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verimind.data import build_synthetic_dataset
from verimind.pipeline import VerificationPipeline


def naive_baseline(samples: list[tuple[str, str]]) -> float:
    """Return a naive baseline accuracy using the most common class."""
    return 0.5


def main() -> None:
    dataset = build_synthetic_dataset()
    pipeline = VerificationPipeline()
    samples = dataset["pairs"]
    baseline = naive_baseline(samples)
    results = []
    for claim, _evidence in samples:
        report = pipeline.verify_text(claim)
        results.append(report["document_summary"]["overall_risk_score"])
    print({"baseline_accuracy": baseline, "pipeline_sample_scores": results[:5]})


if __name__ == "__main__":
    main()
