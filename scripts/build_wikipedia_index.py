"""Build an independent random sample index from the FEVER Wikipedia dump."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verimind.fts_retrieval import build_full_text_index
from verimind.wiki_pages import PASSAGE_WORDS, page_passages

DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
ARCHIVE_URL = "https://fever.ai/download/fever/wiki-pages.zip"
ARCHIVE_PATH = DATA_DIR / "corpus" / "wiki-pages.zip"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def download_archive() -> Path:
    if ARCHIVE_PATH.is_file():
        return ARCHIVE_PATH
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE_PATH.with_suffix(".zip.tmp")
    downloaded = 0
    with requests.get(ARCHIVE_URL, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        expected_size = int(response.headers.get("Content-Length", "0"))
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (256 * 1024 * 1024) < 8 * 1024 * 1024:
                        print(f"Downloaded {downloaded:,} bytes of FEVER Wikipedia dump", flush=True)
    if expected_size and downloaded != expected_size:
        raise RuntimeError(
            f"Incomplete FEVER archive download: received {downloaded} of {expected_size} bytes."
        )
    temporary.replace(ARCHIVE_PATH)
    return ARCHIVE_PATH


def is_usable_page(page: dict[str, Any]) -> bool:
    page_id = page.get("id") or page.get("title")
    text = page.get("text") or page.get("lines")
    return bool(page_id and text and str(text).strip())


def sample_pages_from_archive(
    archive_path: Path, target_pages: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    rows_seen = 0
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
                    if not is_usable_page(page):
                        continue
                    rows_seen += 1
                    if len(sample) < target_pages:
                        sample.append(page)
                    else:
                        replacement = rng.randrange(rows_seen)
                        if replacement < target_pages:
                            sample[replacement] = page
            if rows_seen and rows_seen % 500_000 == 0:
                print(f"Scanned {rows_seen:,} usable pages from the independent dump", flush=True)
    if len(sample) < target_pages:
        raise RuntimeError(
            f"FEVER dump contains only {len(sample)} usable pages; requested {target_pages}."
        )
    return sample, rows_seen


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_index(
    passages: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    *,
    seed: int,
    source_rows_scanned: int,
) -> dict[str, Any]:
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    for start in range(0, len(passages), 128):
        batch = passages[start : start + 128]
        embeddings = model.encode(
            [passage["text"] for passage in batch],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        if embeddings.shape != (len(batch), index.d):
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("The encoder returned non-normalized passage embeddings.")
        index.add(embeddings)

    if index.ntotal != len(passages):
        raise RuntimeError("FAISS index and passage metadata counts differ.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_tmp = INDEX_DIR / "faiss.index.tmp"
    metadata_tmp = INDEX_DIR / "passages_meta.jsonl.tmp"
    manifest_tmp = INDEX_DIR / "index_manifest.json.tmp"
    faiss.write_index(index, str(index_tmp))
    write_jsonl(metadata_tmp, passages)
    manifest = {
        "schema_version": 1,
        "source": "FEVER v1.0 Wikipedia pages dump (2022 snapshot)",
        "source_url": ARCHIVE_URL,
        "sample_method": (
            "reservoir sample from the complete independent Wikipedia page dump; "
            "no evaluation claims or labels used"
        ),
        "sample_seed": seed,
        "source_rows_scanned": source_rows_scanned,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "page_count": len(pages),
        "passage_count": len(passages),
        "embedding_backend": "sentence-transformers",
        "embedding_model": MODEL_NAME,
        "dimension": index.d,
        "normalized_embeddings": True,
        "metric": "inner_product",
        "passage_words": PASSAGE_WORDS,
        "page_extract_character_limit": 40000,
    }
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    index_tmp.replace(INDEX_DIR / "faiss.index")
    metadata_tmp.replace(INDEX_DIR / "passages_meta.jsonl")
    manifest_tmp.replace(INDEX_DIR / "index_manifest.json")
    return manifest


def build_corpus_index(target_pages: int, seed: int) -> dict[str, Any]:
    archive = download_archive()
    pages, source_rows_scanned = sample_pages_from_archive(archive, target_pages, seed)
    all_passages = [
        passage
        for page in pages
        for passage in page_passages(page)
    ]
    if not all_passages:
        raise ValueError("The downloaded Wikipedia sample contains no usable passages.")
    corpus_path = DATA_DIR / "corpus" / "passages.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(corpus_path, all_passages)
    return build_index(
        all_passages,
        pages,
        seed=seed,
        source_rows_scanned=source_rows_scanned,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-text-only",
        action="store_true",
        help="Build a full-corpus SQLite FTS5 index without the sampled FAISS index.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5000,
        help="Number of independently sampled Wikipedia pages to index (default: 5000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reservoir-sampling seed.")
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be positive")
    if args.full_text_only:
        manifest = build_full_text_index(download_archive(), INDEX_DIR / "wiki_fts.sqlite3")
        print(json.dumps(manifest, indent=2))
        return 0
    if not os.environ.get("HF_HUB_OFFLINE"):
        print(
            f"Downloading/loading the embedding model {MODEL_NAME!r} if not cached.",
            file=sys.stderr,
        )
    manifest = build_corpus_index(args.pages, args.seed)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
