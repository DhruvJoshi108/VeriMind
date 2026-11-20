"""Full-corpus SQLite FTS retrieval for the FEVER Wikipedia dump."""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from verimind.faiss_retrieval import informative_terms
from verimind.wiki_pages import page_passages

FTS_TABLE = "wiki_passages"


def build_full_text_index(archive_path: Path, index_path: Path) -> dict[str, Any]:
    """Stream every usable Wikipedia page into an atomically replaced FTS5 database."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"FEVER Wikipedia archive is missing: {archive_path}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_path)
    page_count = 0
    passage_count = 0
    rows_seen = 0
    pending: list[tuple[str, str, int, str, str, str]] = []

    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            f"""
            CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5(
                doc_id UNINDEXED,
                title,
                passage_id UNINDEXED,
                source UNINDEXED,
                domain UNINDEXED,
                text,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        connection.execute(
            "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

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
                        rows_seen += 1
                        if not (page.get("id") or page.get("title")):
                            continue
                        passages = page_passages(page)
                        if not passages:
                            continue

                        page_count += 1
                        for passage in passages:
                            pending.append(
                                (
                                    str(passage["doc_id"]),
                                    str(passage["title"]),
                                    int(passage["passage_id"]),
                                    str(passage["source"]),
                                    str(passage["domain"]),
                                    str(passage["text"]),
                                )
                            )
                        if len(pending) >= 2000:
                            connection.executemany(
                                f"INSERT INTO {FTS_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                                pending,
                            )
                            passage_count += len(pending)
                            pending.clear()
                            connection.commit()
                        if rows_seen % 500_000 == 0:
                            print(
                                f"Scanned {rows_seen:,} Wikipedia rows; "
                                f"indexed {passage_count + len(pending):,} passages",
                                file=sys.stderr,
                                flush=True,
                            )

        if pending:
            connection.executemany(
                f"INSERT INTO {FTS_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                pending,
            )
            passage_count += len(pending)
        if passage_count == 0:
            raise ValueError(f"No usable Wikipedia passages found in {archive_path}")

        metadata = {
            "schema_version": 1,
            "source": "FEVER v1.0 Wikipedia pages dump (2022 snapshot)",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "page_count": page_count,
            "passage_count": passage_count,
            "retrieval_backend": "sqlite-fts5",
        }
        connection.executemany(
            "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )
        connection.commit()
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    temporary_path.replace(index_path)
    return metadata


class SQLiteFtsEvidenceRetriever:
    """Retrieve passages from a streamed full-corpus SQLite FTS5 index."""

    def __init__(self, index_path: Path) -> None:
        if not index_path.is_file():
            raise FileNotFoundError(f"SQLite FTS index is missing: {index_path}")
        self.index_path = index_path
        with closing(sqlite3.connect(index_path)) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (FTS_TABLE,),
            ).fetchone()
        if exists is None:
            raise ValueError(f"SQLite FTS index has no {FTS_TABLE!r} table: {index_path}")

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be positive")
        terms = sorted(informative_terms(query))
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms)

        with closing(sqlite3.connect(self.index_path)) as connection:
            rows = connection.execute(
                f"""
                SELECT doc_id, title, passage_id, source, domain, text,
                       bm25({FTS_TABLE}, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0) AS rank
                FROM {FTS_TABLE}
                WHERE {FTS_TABLE} MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_query, max(k * 10, 50)),
            ).fetchall()

        results = []
        seen_documents: set[str] = set()
        for doc_id, title, passage_id, source, domain, text, rank in rows:
            doc_id = str(doc_id)
            if doc_id in seen_documents:
                continue
            seen_documents.add(doc_id)
            relevance = max(0.0, -float(rank))
            results.append(
                {
                    "doc_id": doc_id,
                    "title": str(title),
                    "passage_id": int(passage_id),
                    "source": str(source),
                    "domain": str(domain),
                    "text": str(text),
                    "score": relevance / (1.0 + relevance),
                }
            )
            if len(results) == k:
                break
        return results
