"""Evaluate the lightweight NLI model on a synthetic test split."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verimind.data import build_synthetic_dataset
from verimind.nli_model import NLIModel


def main() -> None:
    dataset = build_synthetic_dataset()
    pairs = dataset["pairs"]
    labels = dataset["labels"]
    # Keep one example per class for a deterministic, representative holdout.
    test_indices = {next(index for index, label in enumerate(labels) if label == target) for target in set(labels)}
    train_pairs = [pair for index, pair in enumerate(pairs) if index not in test_indices]
    train_labels = [label for index, label in enumerate(labels) if index not in test_indices]
    test_pairs = [pair for index, pair in enumerate(pairs) if index in test_indices]
    test_labels = [label for index, label in enumerate(labels) if index in test_indices]

    model = NLIModel().fit(train_pairs, train_labels)
    predictions = model.predict(test_pairs)
    accuracy = sum(pred == label for pred, label in zip(predictions, test_labels, strict=True)) / len(test_labels)
    print({"accuracy": accuracy, "predictions": predictions, "labels": test_labels})


if __name__ == "__main__":
    main()
