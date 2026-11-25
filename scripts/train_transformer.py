"""Fine-tune the local NLI transformer on leakage-safe prepared FEVER pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import split_by_claim
from verimind.transformer_nli import DEFAULT_MODEL_NAME, TransformerNLIModel

LABELS = ("entailment", "contradiction", "neutral")


def normalize_claim(claim: str) -> str:
    return " ".join(claim.casefold().split())


def load_training_examples(
    path: Path,
    *,
    max_per_label: int,
    seed: int,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            claim = row.get("claim")
            evidence = row.get("evidence")
            label = row.get("label")
            if (
                not isinstance(claim, str)
                or not claim.strip()
                or not isinstance(evidence, str)
                or not evidence.strip()
                or label not in LABELS
            ):
                raise ValueError(f"Invalid claim/evidence/label at {path}:{line_number}")
            examples.append(
                {
                    "claim": claim.strip(),
                    "evidence": evidence.strip(),
                    "label": label,
                    "claim_key": normalize_claim(claim),
                }
            )

    by_label = {label: [row for row in examples if row["label"] == label] for label in LABELS}
    if any(not rows for rows in by_label.values()):
        raise ValueError(f"Training data must include all labels: {', '.join(LABELS)}")
    if max_per_label:
        rng = random.Random(seed)
        examples = [
            row
            for label in LABELS
            for row in rng.sample(by_label[label], min(max_per_label, len(by_label[label])))
        ]
    return examples


def split_examples(
    examples: list[dict[str, str]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int, int]:
    pairs = [(row["claim"], row["evidence"]) for row in examples]
    labels = [row["label"] for row in examples]
    claim_groups = [row["claim_key"] for row in examples]
    (
        train_pairs,
        validation_pairs,
        train_labels,
        validation_labels,
        train_claims,
        validation_claims,
    ) = split_by_claim(pairs, labels, claim_groups, validation_fraction, seed)
    train_rows = [
        {"claim": claim, "evidence": evidence, "label": label}
        for (claim, evidence), label in zip(train_pairs, train_labels, strict=True)
    ]
    validation_rows = [
        {"claim": claim, "evidence": evidence, "label": label}
        for (claim, evidence), label in zip(
            validation_pairs, validation_labels, strict=True
        )
    ]
    return train_rows, validation_rows, train_claims, validation_claims


class PairDataset:
    def __init__(self, examples: list[dict[str, str]], label_to_id: dict[str, int]) -> None:
        self.examples = examples
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, str, int]:
        row = self.examples[index]
        return row["claim"], row["evidence"], self.label_to_id[row["label"]]


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    use_amp = device.type == "cuda"

    examples = load_training_examples(
        args.data_path,
        max_per_label=args.max_per_label,
        seed=args.seed,
    )
    train_rows, validation_rows, train_claims, validation_claims = split_examples(
        examples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    labels_by_id = TransformerNLIModel._ordered_labels(model.config.id2label)
    label_to_id = {label: index for index, label in enumerate(labels_by_id)}
    model.to(device)

    def collate(batch: list[tuple[str, str, int]]) -> dict[str, Any]:
        claims, evidence, label_ids = zip(*batch, strict=True)
        encoded = tokenizer(
            list(claims),
            list(evidence),
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return encoded

    train_loader = DataLoader(
        PairDataset(train_rows, label_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=use_amp,
    )
    validation_loader = DataLoader(
        PairDataset(validation_rows, label_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=use_amp,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = max(1, int(total_updates * 0.1))

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    args.output.mkdir(parents=True, exist_ok=True)

    print(
        f"Training {len(train_rows):,} pairs from {train_claims:,} claims; "
        f"validating on {len(validation_rows):,} pairs from {validation_claims:,} claims "
        f"using {device}."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        processed = 0
        for batch_index, batch in enumerate(train_loader):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            group_start = batch_index // args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps,
                len(train_loader) - group_start * args.gradient_accumulation_steps,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**batch)
                loss = output.loss
            scaler.scale(loss / group_size).backward()
            total_loss += float(loss.detach()) * batch["labels"].size(0)
            processed += batch["labels"].size(0)

            last_batch = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % args.gradient_accumulation_steps == 0 or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs}: "
                    f"{batch_index + 1}/{len(train_loader)} batches",
                    flush=True,
                )

        model.eval()
        validation_loss = 0.0
        correct = 0
        validation_count = 0
        with torch.inference_mode():
            for batch in validation_loader:
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    output = model(**batch)
                count = batch["labels"].size(0)
                validation_loss += float(output.loss) * count
                correct += int((output.logits.argmax(dim=-1) == batch["labels"]).sum())
                validation_count += count

        epoch_metrics = {
            "epoch": epoch,
            "training_loss": total_loss / max(1, processed),
            "validation_loss": validation_loss / max(1, validation_count),
            "validation_accuracy": correct / max(1, validation_count),
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)

        if epoch_metrics["validation_accuracy"] > best_accuracy:
            best_accuracy = epoch_metrics["validation_accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    report = {
        "base_model": args.model_name,
        "training_data": str(args.data_path),
        "max_per_label": args.max_per_label or None,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_claims": train_claims,
        "validation_claims": validation_claims,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "history": history,
        "saved_model": str(args.output.resolve()),
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "corpus" / "fever_nli_train.jsonl",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "deberta_fever_finetuned",
    )
    parser.add_argument("--max-per-label", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.data_path.is_file():
        parser.error(f"Training data not found: {args.data_path}")
    if args.max_per_label < 0:
        parser.error("--max-per-label must be zero (unlimited) or positive")
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.max_length < 1:
        parser.error("batch size, gradient accumulation, and max length must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be a positive finite number")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error(f"Output directory is not empty: {args.output}")

    print(json.dumps(train(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
