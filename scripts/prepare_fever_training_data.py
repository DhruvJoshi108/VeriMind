"""Prepare leakage-safe FEVER claim/evidence pairs for local NLI training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verimind.features import token_overlap

DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
ARCHIVE_PATH = CORPUS_DIR / "wiki-pages.zip"
TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"
TRAIN_PATH = CORPUS_DIR / "fever_train.jsonl"
OUTPUT_PATH = CORPUS_DIR / "fever_nli_train.jsonl"
EVAL_CASES_PATH = DATA_DIR / "eval_cases.jsonl"
PASSAGES_PATH = CORPUS_DIR / "passages.jsonl"
LABEL_MAP = {
    "SUPPORTS": "entailment",
    "REFUTES": "contradiction",
    "NOT ENOUGH INFO": "neutral",
}


@dataclass(frozen=True)
class FeverClaim:
    claim_id: str
    claim: str
    label: str
    evidence_groups: tuple[tuple[tuple[str, int], ...], ...]


def normalize_claim(claim: str) -> str:
    return " ".join(claim.casefold().split())


def normalize_title(title: str) -> str:
    return " ".join(title.replace("_", " ").casefold().split())


def evidence_groups_for(row: dict[str, Any]) -> tuple[tuple[tuple[str, int], ...], ...]:
    groups = []
    for group in row.get("evidence") or []:
        references = []
        for item in group or []:
            if not isinstance(item, list) or len(item) < 4 or not item[2]:
                continue
            try:
                line_number = int(item[3])
            except (TypeError, ValueError):
                continue
            if line_number >= 0:
                references.append((normalize_title(str(item[2])), line_number))
        unique_references = tuple(dict.fromkeys(references))
        if unique_references:
            groups.append(unique_references)
    return tuple(groups)


def _id_variants(value: Any) -> set[str]:
    identifier = str(value)
    variants = {identifier}
    if identifier.startswith("fever-"):
        variants.add(identifier.removeprefix("fever-"))
    return variants


def read_holdouts(path: Path) -> tuple[set[str], set[str]]:
    heldout_ids: set[str] = set()
    heldout_claims: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            case = json.loads(line)
            heldout_ids.update(_id_variants(case.get("id", "")))
            claim = str(case.get("claim_or_answer", "")).strip()
            if claim:
                heldout_claims.add(normalize_claim(claim))
    return heldout_ids, heldout_claims


def download_training_data(path: Path) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    downloaded = 0
    with requests.get(TRAIN_URL, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        expected_size = int(response.headers.get("Content-Length", "0"))
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (16 * 1024 * 1024) < 4 * 1024 * 1024:
                        print(f"Downloaded {downloaded:,} FEVER training bytes", flush=True)
    if expected_size and downloaded != expected_size:
        raise RuntimeError(
            f"Incomplete FEVER train download: received {downloaded} of {expected_size} bytes."
        )
    temporary.replace(path)
    return path


def collect_claims(
    train_path: Path,
    heldout_ids: set[str],
    heldout_claims: set[str],
) -> tuple[list[FeverClaim], list[FeverClaim], dict[str, set[int]], dict[str, int]]:
    verifiable: list[FeverClaim] = []
    unverifiable: list[FeverClaim] = []
    required_lines: dict[str, set[int]] = {}
    seen_claims: set[str] = set()
    counts = {"source_rows": 0, "heldout_rows_excluded": 0, "duplicate_claims_skipped": 0}

    with train_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["source_rows"] += 1
            claim = str(row.get("claim", "")).strip()
            claim_id = str(row.get("id", ""))
            if not claim:
                continue
            claim_key = normalize_claim(claim)
            if (
                _id_variants(claim_id) & heldout_ids
                or claim_key in heldout_claims
            ):
                counts["heldout_rows_excluded"] += 1
                continue
            if claim_key in seen_claims:
                counts["duplicate_claims_skipped"] += 1
                continue
            seen_claims.add(claim_key)

            label = LABEL_MAP.get(str(row.get("label", "")).upper())
            if label == "neutral":
                unverifiable.append(FeverClaim(claim_id, claim, label, ()))
                continue
            if label not in {"entailment", "contradiction"}:
                continue

            groups = evidence_groups_for(row)
            if not groups:
                continue
            verifiable.append(FeverClaim(claim_id, claim, label, groups))
            for group in groups:
                for title, line_number in group:
                    required_lines.setdefault(title, set()).add(line_number)

    return verifiable, unverifiable, required_lines, counts


def extract_evidence_lines(
    archive_path: Path,
    required_lines: dict[str, set[int]],
) -> tuple[dict[tuple[str, int], str], int, int]:
    pending = {title: set(lines) for title, lines in required_lines.items()}
    found: dict[tuple[str, int], str] = {}
    rows_scanned = 0
    with ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("wiki-pages/")
            and not name.startswith("__MACOSX/")
            and not Path(name).name.startswith("._")
            and name.lower().endswith((".jsonl", ".json"))
        )
        if not members:
            raise ValueError(f"No JSON/JSONL page files found in {archive_path}")
        for member in members:
            with archive.open(member) as source:
                for line in source:
                    if not line.strip():
                        continue
                    page = json.loads(line)
                    rows_scanned += 1
                    page_id = str(page.get("id") or page.get("title") or "")
                    title = normalize_title(page_id)
                    if title not in pending:
                        continue
                    raw_lines = page.get("lines") or page.get("text") or ""
                    page_lines = (
                        raw_lines
                        if isinstance(raw_lines, list)
                        else str(raw_lines).splitlines()
                    )
                    for raw_line in page_lines:
                        line_number_text, separator, sentence = str(raw_line).partition("\t")
                        if not separator:
                            continue
                        try:
                            line_number = int(line_number_text)
                        except ValueError:
                            continue
                        if line_number in pending[title] and sentence.strip():
                            found[(title, line_number)] = sentence.strip()
                            pending[title].discard(line_number)
                    if not pending[title]:
                        del pending[title]
                    if not pending:
                        break
            if rows_scanned and rows_scanned % 500_000 == 0:
                print(
                    f"Scanned {rows_scanned:,} Wikipedia rows; "
                    f"found {len(found):,} annotated evidence lines",
                    flush=True,
                )
            if not pending:
                break
    missing_lines = sum(len(lines) for lines in pending.values())
    return found, rows_scanned, missing_lines


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_passage_pool(path: Path) -> list[str]:
    passages = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            text = str(json.loads(line).get("text", "")).strip()
            if text:
                passages.append(text)
    if not passages:
        raise ValueError(f"No passages found in {path}")
    return passages


def prepare_examples(
    verifiable: list[FeverClaim],
    unverifiable: list[FeverClaim],
    evidence_lines: dict[tuple[str, int], str],
    passages_path: Path,
    max_per_label: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_label: dict[str, list[dict[str, Any]]] = {
        "entailment": [],
        "contradiction": [],
        "neutral": [],
    }
    rng = random.Random(seed)
    missing_groups = 0
    for record in verifiable:
        for group in record.evidence_groups:
            if any(reference not in evidence_lines for reference in group):
                missing_groups += 1
                continue
            sentences = list(dict.fromkeys(evidence_lines[reference] for reference in group))
            evidence = " ".join(sentences)
            if evidence:
                by_label[record.label].append(
                    {
                        "claim_id": record.claim_id,
                        "claim_key": normalize_claim(record.claim),
                        "claim": record.claim,
                        "evidence": evidence,
                        "label": record.label,
                        "source": "FEVER v1.0 train",
                    }
                )

    passage_pool = load_passage_pool(passages_path)
    neutral_without_low_overlap_passage = 0
    neutral_claims = unverifiable
    if max_per_label and len(neutral_claims) > max_per_label:
        neutral_claims = rng.sample(neutral_claims, max_per_label)
    for record in neutral_claims:
        evidence = None
        for _ in range(50):
            candidate = rng.choice(passage_pool)
            if token_overlap(record.claim, candidate) <= 0.08:
                evidence = candidate
                break
        if evidence is None:
            neutral_without_low_overlap_passage += 1
            continue
        by_label["neutral"].append(
            {
                "claim_id": record.claim_id,
                "claim_key": normalize_claim(record.claim),
                "claim": record.claim,
                "evidence": evidence,
                "label": "neutral",
                "source": "FEVER v1.0 train + random low-overlap Wikipedia passage (weak neutral pair)",
            }
        )

    for label, rows in by_label.items():
        if max_per_label and len(rows) > max_per_label:
            by_label[label] = rng.sample(rows, max_per_label)
    examples = [row for label in by_label.values() for row in label]
    return examples, {
        "evidence_groups_missing_lines": missing_groups,
        "neutral_claims_without_low_overlap_passage": neutral_without_low_overlap_passage,
        "examples_by_label": {label: len(rows) for label, rows in by_label.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-label", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_per_label < 0:
        parser.error("--max-per-label must be zero (unlimited) or positive")
    if not ARCHIVE_PATH.is_file():
        parser.error(f"FEVER Wikipedia archive is missing: {ARCHIVE_PATH}")
    if not PASSAGES_PATH.is_file():
        parser.error(f"Sampled Wikipedia passages are missing: {PASSAGES_PATH}")
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        parser.error(f"Output already exists; pass --overwrite to replace it: {args.output}")

    train_path = download_training_data(TRAIN_PATH)
    heldout_ids, heldout_claims = read_holdouts(EVAL_CASES_PATH)
    verifiable, unverifiable, required_lines, counts = collect_claims(
        train_path,
        heldout_ids,
        heldout_claims,
    )
    evidence_lines, wiki_rows_scanned, missing_evidence_lines = extract_evidence_lines(
        ARCHIVE_PATH,
        required_lines,
    )
    examples, preparation_counts = prepare_examples(
        verifiable,
        unverifiable,
        evidence_lines,
        PASSAGES_PATH,
        args.max_per_label,
        args.seed,
    )
    if not examples or set(row["label"] for row in examples) != set(LABEL_MAP.values()):
        raise RuntimeError("Prepared training data must contain all three NLI labels.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output, examples)
    manifest = {
        "source": "FEVER v1.0 train.jsonl with FEVER v1.0 Wikipedia pages",
        "source_url": TRAIN_URL,
        "seed": args.seed,
        "heldout_dev_test_claims_excluded": len(heldout_claims),
        "fever_rows": counts,
        "wiki_rows_scanned": wiki_rows_scanned,
        "unique_evidence_lines_found": len(evidence_lines),
        "evidence_lines_missing_from_wikipedia_snapshot": missing_evidence_lines,
        **preparation_counts,
        "max_per_label": args.max_per_label or None,
        "weak_neutral_pair_method": (
            "random low-overlap passage from the independent FEVER Wikipedia page sample "
            "for NOT ENOUGH INFO claims"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(args.output.resolve()), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
