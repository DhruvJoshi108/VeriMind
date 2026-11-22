import json
from zipfile import ZipFile

from scripts.prepare_fever_training_data import collect_claims, extract_evidence_lines
from scripts.train import split_by_claim


def test_collect_claims_excludes_heldout_ids_and_normalized_claims(tmp_path):
    rows = [
        {
            "id": 1,
            "claim": "A supported claim.",
            "label": "SUPPORTS",
            "evidence": [[[0, 10, "Sample_Page", 2]]],
        },
        {"id": 2, "claim": "An unknown claim.", "label": "NOT ENOUGH INFO", "evidence": []},
        {
            "id": 100,
            "claim": "A held-out claim.",
            "label": "REFUTES",
            "evidence": [[[0, 11, "Other_Page", 3]]],
        },
        {
            "id": 101,
            "claim": " A HELD-OUT   claim. ",
            "label": "SUPPORTS",
            "evidence": [[[0, 12, "Other_Page", 4]]],
        },
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    verifiable, unverifiable, required, counts = collect_claims(
        source,
        {"100"},
        {"a held-out claim."},
    )

    assert [claim.claim_id for claim in verifiable] == ["1"]
    assert [claim.claim_id for claim in unverifiable] == ["2"]
    assert required == {"sample page": {2}}
    assert counts["heldout_rows_excluded"] == 2


def test_extract_evidence_lines_joins_requested_wiki_lines(tmp_path):
    archive_path = tmp_path / "wiki.zip"
    page = {
        "id": "Sample_Page",
        "lines": "0\tTitle text\n2\tEvidence sentence.\n",
    }
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("wiki-pages/sample.jsonl", json.dumps(page) + "\n")

    lines, rows_scanned, missing = extract_evidence_lines(
        archive_path,
        {"sample page": {2, 4}},
    )

    assert lines == {("sample page", 2): "Evidence sentence."}
    assert rows_scanned == 1
    assert missing == 1


def test_fever_validation_split_keeps_all_claim_evidence_pairs_together():
    pairs = [(f"claim {index}", f"evidence {pair}") for index in range(12) for pair in range(2)]
    labels = [("entailment", "contradiction", "neutral")[index % 3] for index in range(12) for _ in range(2)]
    groups = [f"claim {index}" for index in range(12) for _ in range(2)]

    train_pairs, validation_pairs, train_labels, validation_labels, train_count, validation_count = (
        split_by_claim(pairs, labels, groups, 0.25, 42)
    )

    train_claims = {claim for claim, _ in train_pairs}
    validation_claims = {claim for claim, _ in validation_pairs}
    assert train_claims.isdisjoint(validation_claims)
    assert len(train_pairs) == 2 * train_count
    assert len(validation_pairs) == 2 * validation_count
    assert set(train_labels) == set(validation_labels) == {
        "entailment",
        "contradiction",
        "neutral",
    }
