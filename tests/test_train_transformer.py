import json
from collections import Counter

from scripts.train_transformer import load_training_examples, split_examples


def test_training_loader_caps_each_label_deterministically(tmp_path):
    path = tmp_path / "training.jsonl"
    rows = [
        {
            "claim": f"{label} claim {index}",
            "evidence": f"Evidence {index}",
            "label": label,
        }
        for label in ("entailment", "contradiction", "neutral")
        for index in range(3)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = load_training_examples(path, max_per_label=2, seed=42)

    assert len(selected) == 6
    assert Counter(row["label"] for row in selected) == {
        "entailment": 2,
        "contradiction": 2,
        "neutral": 2,
    }
    assert selected == load_training_examples(path, max_per_label=2, seed=42)


def test_transformer_training_split_keeps_claims_separate():
    rows = [
        {
            "claim": f"{label} claim {claim_index}",
            "evidence": f"Evidence {evidence_index}",
            "label": label,
            "claim_key": f"{label} claim {claim_index}",
        }
        for label in ("entailment", "contradiction", "neutral")
        for claim_index in range(5)
        for evidence_index in range(2)
    ]

    train, validation, train_claims, validation_claims = split_examples(
        rows,
        validation_fraction=0.2,
        seed=42,
    )

    train_groups = {row["claim"] for row in train}
    validation_groups = {row["claim"] for row in validation}
    assert train_groups.isdisjoint(validation_groups)
    assert train_claims == 12
    assert validation_claims == 3
