"""Compare the local NLI model with a majority-class baseline on synthetic data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verimind.data import build_synthetic_dataset
from verimind.nli_model import NLIModel


def main() -> None:
    """Report held-out accuracy; this synthetic benchmark is not a quality claim."""
    dataset = build_synthetic_dataset()
    pairs = dataset["pairs"]
    labels = dataset["labels"]
    train_indices, test_indices = train_test_split(
        list(range(len(labels))),
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    train_pairs = [pairs[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    test_pairs = [pairs[index] for index in test_indices]
    test_labels = [labels[index] for index in test_indices]

    model = NLIModel().fit(train_pairs, train_labels)
    predictions = model.predict(test_pairs)
    majority_label = Counter(train_labels).most_common(1)[0][0]
    model_accuracy = sum(
        predicted == actual for predicted, actual in zip(predictions, test_labels, strict=True)
    ) / len(test_labels)
    baseline_accuracy = sum(label == majority_label for label in test_labels) / len(test_labels)

    print(
        json.dumps(
            {
                "evaluation": "synthetic_nli_holdout",
                "warning": "Small synthetic benchmark; not representative of real-world accuracy.",
                "n_test": len(test_labels),
                "majority_class": majority_label,
                "majority_baseline_accuracy": baseline_accuracy,
                "nli_model_accuracy": model_accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
