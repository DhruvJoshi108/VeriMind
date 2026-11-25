"""SQLite and PostgreSQL persistence for reports and evidence."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import URL, Connection, Engine, make_url

from verimind.config import settings

metadata = MetaData()
reports = Table(
    "reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("document_text", Text, nullable=False),
    Column("summary_json", Text, nullable=False),
)
claims = Table(
    "claims",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("report_id", Integer, ForeignKey("reports.id"), nullable=False),
    Column("claim", Text, nullable=False),
    Column("verdict", String, nullable=False),
    Column("risk_score", Float, nullable=False),
    Column("details_json", Text, nullable=False),
)
evidence = Table(
    "evidence",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("claim_id", Integer, ForeignKey("claims.id"), nullable=False),
    Column("text", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("domain", String, nullable=False),
    Column("score", Float, nullable=False),
)
verdicts = Table(
    "verdicts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("report_id", Integer, ForeignKey("reports.id"), nullable=False),
    Column("verdict", String, nullable=False),
    Column("risk_score", Float, nullable=False),
    Column("why", Text, nullable=False),
)


def get_db_path() -> Path:
    """Return the local SQLite file location used when DATABASE_URL is unset."""
    return Path(os.getenv("VERIMIND_DB_PATH", "data/verimind.db"))


def _build_database_url() -> URL:
    configured_url = settings.database_url
    if not configured_url:
        return URL.create("sqlite", database=str(get_db_path().resolve()))

    if configured_url.startswith("postgres://"):
        configured_url = "postgresql+psycopg://" + configured_url.removeprefix("postgres://")
    elif configured_url.startswith("postgresql://"):
        configured_url = "postgresql+psycopg://" + configured_url.removeprefix("postgresql://")
    return make_url(configured_url)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and reuse the configured database engine."""
    url = _build_database_url()
    connect_args: dict[str, Any] = {}
    if url.get_backend_name() == "sqlite":
        connect_args["check_same_thread"] = False
        if url.database and url.database != ":memory:":
            Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def init_db() -> Connection:
    """Create required tables if absent and return an open connection."""
    engine = get_engine()
    metadata.create_all(engine)
    return engine.connect()


def check_database() -> None:
    """Verify database connectivity and create the report schema if needed."""
    conn = init_db()
    try:
        conn.execute(text("SELECT 1"))
    finally:
        conn.close()


def save_report(document_text: str, report: dict[str, Any]) -> int:
    """Persist a report and return its identifier."""
    conn = init_db()
    try:
        with conn.begin():
            result = conn.execute(
                insert(reports).values(
                    created_at=func.current_timestamp(),
                    document_text=document_text,
                    summary_json=json.dumps(report, ensure_ascii=False, default=str),
                )
            )
            report_id = int(result.inserted_primary_key[0])
            for claim in report.get("claims", []):
                claim_result = conn.execute(
                    insert(claims).values(
                        report_id=report_id,
                        claim=claim.get("claim", ""),
                        verdict=claim.get("verdict", "uncertain"),
                        risk_score=claim.get("risk_score", 0.0),
                        details_json=json.dumps(claim, ensure_ascii=False, default=str),
                    )
                )
                claim_id = int(claim_result.inserted_primary_key[0])
                for item in claim.get("evidence", []):
                    conn.execute(
                        insert(evidence).values(
                            claim_id=claim_id,
                            text=item.get("text", ""),
                            source=item.get("source", "unknown"),
                            domain=item.get("domain", "unknown"),
                            score=item.get("score", 0.0),
                        )
                    )
    finally:
        conn.close()
    return report_id


def get_report(report_id: int) -> dict[str, Any] | None:
    """Load one stored report by id."""
    conn = init_db()
    try:
        row = conn.execute(
            select(reports.c.summary_json).where(reports.c.id == report_id)
        ).first()
    finally:
        conn.close()
    if row is None:
        return None
    return json.loads(row[0])
