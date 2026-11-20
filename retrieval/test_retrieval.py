"""Smoke test for relevant and unrelated queries against the FAISS index."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verimind.faiss_retrieval import FaissEvidenceRetriever

INDEX_DIR = ROOT / "data/index"


def main() -> None:
    retriever = FaissEvidenceRetriever.from_index_dir(INDEX_DIR)
    first = retriever.passages[0]
    second = next(
        passage
        for passage in retriever.passages[1:]
        if passage["doc_id"] != first["doc_id"]
    )
    first_results = retriever.search(" ".join(first["text"].split()[:40]))
    second_results = retriever.search(" ".join(second["text"].split()[:40]))
    unrelated_results = retriever.search("qzxv unmatchedtestterm")

    if not first_results or first_results[0]["doc_id"] != first["doc_id"]:
        raise AssertionError("The first passage query did not retrieve its source page.")
    if not second_results or second_results[0]["doc_id"] != second["doc_id"]:
        raise AssertionError("The second passage query did not retrieve its source page.")
    if first_results[0]["doc_id"] == second_results[0]["doc_id"]:
        raise AssertionError("Distinct queries retrieved the same top passage.")
    if unrelated_results:
        raise AssertionError("The unsupported query returned unrelated evidence.")

    for result in first_results[:1] + second_results[:1]:
        print(f"{result['score']:.4f} | {result['title']} | {result['text']}")
    print("PASS: unrelated query returned no evidence.")


if __name__ == "__main__":
    main()
