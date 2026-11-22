"""Train the model on a synthetic claim-evidence dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verimind.data import build_synthetic_dataset
from verimind.nli_model import NLIModel

LABELS = ("entailment", "contradiction", "neutral")


def split_by_claim(
    pairs: list[tuple[str, str]],
    labels: list[str],
    claim_groups: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str], list[str], int, int]:
    group_labels: dict[str, str] = {}
    for group, label in zip(claim_groups, labels, strict=True):
        if group in group_labels and group_labels[group] != label:
            raise ValueError("A claim appears with conflicting labels in FEVER training data.")
        group_labels[group] = label
    train_groups, validation_groups = train_test_split(
        list(group_labels),
        test_size=validation_fraction,
        random_state=seed,
        stratify=list(group_labels.values()),
    )
    train_group_set = set(train_groups)
    validation_group_set = set(validation_groups)
    train_indices = [i for i, group in enumerate(claim_groups) if group in train_group_set]
    validation_indices = [
        i for i, group in enumerate(claim_groups) if group in validation_group_set
    ]
    return (
        [pairs[i] for i in train_indices],
        [pairs[i] for i in validation_indices],
        [labels[i] for i in train_indices],
        [labels[i] for i in validation_indices],
        len(train_group_set),
        len(validation_group_set),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "fever"),
        default="synthetic",
        help="Training source; FEVER requires scripts/prepare_fever_training_data.py first.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "corpus" / "fever_nli_train.jsonl",
        help="Prepared JSONL path when --dataset=fever.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=0,
        help="For FEVER training, use at most this many pairs per class; zero uses all pairs.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=12, help="Early-stopping patience.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.25,
        help="Fraction of the synthetic examples held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Stratified split seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "nli_model.pkl",
        help="Path for the trained model artifact.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.patience < 1:
        parser.error("--patience must be positive")
    if args.max_per_label < 0:
        parser.error("--max-per-label must be zero (unlimited) or positive")
    if args.dataset == "synthetic" and args.max_per_label:
        parser.error("--max-per-label is only supported with --dataset=fever")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    if args.dataset == "synthetic":
        dataset = build_synthetic_dataset()
        pairs = dataset["pairs"]
        labels = dataset["labels"]
        train_pairs, validation_pairs, train_labels, validation_labels = train_test_split(
            pairs,
            labels,
            test_size=args.validation_fraction,
            random_state=args.seed,
            stratify=labels,
        )
        train_claim_count = validation_claim_count = None
        training_source = "synthetic (42 claim-evidence pairs)"
    else:
        if not args.data_path.is_file():
            parser.error(
                f"Prepared FEVER data not found: {args.data_path}. "
                "Run scripts/prepare_fever_training_data.py first."
            )
        rows = [
            json.loads(line)
            for line in args.data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prepared_example_count = len(rows)
        if args.max_per_label:
            rng = random.Random(args.seed)
            rows_by_label: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                rows_by_label.setdefault(str(row["label"]), []).append(row)
            rows = [
                row
                for label in sorted(rows_by_label)
                for row in rng.sample(
                    rows_by_label[label],
                    min(args.max_per_label, len(rows_by_label[label])),
                )
            ]
        pairs = [(str(row["claim"]), str(row["evidence"])) for row in rows]
        labels = [str(row["label"]) for row in rows]
        claim_groups = [" ".join(str(row["claim"]).casefold().split()) for row in rows]
        (
            train_pairs,
            validation_pairs,
            train_labels,
            validation_labels,
            train_claim_count,
            validation_claim_count,
        ) = split_by_claim(
            pairs,
            labels,
            claim_groups,
            args.validation_fraction,
            args.seed,
        )
        training_source = f"FEVER v1.0 ({len(rows):,} selected from {prepared_example_count:,} pairs)"

    model = NLIModel()
    history = model.fit_with_validation(
        train_pairs,
        train_labels,
        validation_pairs,
        validation_labels,
        epochs=args.epochs,
        patience=args.patience,
    )
    predictions = model.predict(validation_pairs)
    model.save(args.output)
    best_epoch = max(history, key=lambda row: row["validation_accuracy"])["epoch"]
    print(
        json.dumps(
            {
                "training_data": training_source,
                "max_per_label": args.max_per_label or None,
                "train_examples": len(train_pairs),
                "validation_examples": len(validation_pairs),
                "train_claims": train_claim_count,
                "validation_claims": validation_claim_count,
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_validation_accuracy": accuracy_score(validation_labels, predictions),
                "classification_report": classification_report(
                    validation_labels,
                    predictions,
                    labels=LABELS,
                    target_names=LABELS,
                    zero_division=0,
                    output_dict=True,
                ),
                "epoch_history": history,
                "saved_model": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
