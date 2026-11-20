"""SQLite-backed persistence for reports and evidence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def get_db_path() -> Path:
    """Return the SQLite file location."""
    return Path("data/verimind.db")


def init_db() -> sqlite3.Connection:
    """Create required tables if absent."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            document_text TEXT NOT NULL,
            summary_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL,
            claim TEXT NOT NULL,
            verdict TEXT NOT NULL,
            risk_score REAL NOT NULL,
            details_json TEXT NOT NULL,
            FOREIGN KEY(report_id) REFERENCES reports(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            domain TEXT NOT NULL,
            score REAL NOT NULL,
            FOREIGN KEY(claim_id) REFERENCES claims(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verdicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            risk_score REAL NOT NULL,
            why TEXT NOT NULL,
            FOREIGN KEY(report_id) REFERENCES reports(id)
        )
        """
    )
    conn.commit()
    return conn


def save_report(document_text: str, report: dict[str, Any]) -> int:
    """Persist a report and return its identifier."""
    conn = init_db()
    cursor = conn.execute(
        "INSERT INTO reports(created_at, document_text, summary_json) VALUES (datetime('now'), ?, ?)",
        (document_text, json.dumps(report, ensure_ascii=False, default=str)),
    )
    report_id = int(cursor.lastrowid)
    for claim in report.get("claims", []):
        claim_cursor = conn.execute(
            "INSERT INTO claims(report_id, claim, verdict, risk_score, details_json) VALUES (?, ?, ?, ?, ?)",
            (
                report_id,
                claim.get("claim", ""),
                claim.get("verdict", "uncertain"),
                claim.get("risk_score", 0.0),
                json.dumps(claim, ensure_ascii=False, default=str),
            ),
        )
        claim_id = int(claim_cursor.lastrowid)
        for evidence in claim.get("evidence", []):
            conn.execute(
                "INSERT INTO evidence(claim_id, text, source, domain, score) VALUES (?, ?, ?, ?, ?)",
                (
                    claim_id,
                    evidence.get("text", ""),
                    evidence.get("source", "unknown"),
                    evidence.get("domain", "unknown"),
                    evidence.get("score", 0.0),
                ),
            )
    conn.commit()
    conn.close()
    return report_id


def get_report(report_id: int) -> dict[str, Any] | None:
    """Load one stored report by id."""
    conn = init_db()
    row = conn.execute(
        "SELECT summary_json FROM reports WHERE id = ?",
        (report_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row[0])
