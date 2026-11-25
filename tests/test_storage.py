from dataclasses import replace

from sqlalchemy.engine import make_url

from verimind import storage


def test_report_round_trip_uses_configured_sqlite_database(tmp_path, monkeypatch):
    database_path = tmp_path / "nested" / "verimind.db"
    monkeypatch.setattr(
        storage,
        "settings",
        replace(storage.settings, database_url=f"sqlite:///{database_path.as_posix()}"),
    )
    storage.get_engine.cache_clear()
    report = {
        "document_summary": {"total_claims": 1},
        "claims": [
            {
                "claim": "A test claim",
                "verdict": "uncertain",
                "risk_score": 50.0,
                "evidence": [
                    {
                        "text": "Test evidence",
                        "source": "https://example.com",
                        "domain": "example.com",
                        "score": 0.8,
                    }
                ],
            }
        ],
    }

    try:
        report_id = storage.save_report("A test claim", report)

        assert storage.get_report(report_id) == report
        assert database_path.is_file()
    finally:
        storage.get_engine().dispose()
        storage.get_engine.cache_clear()


def test_postgres_urls_use_psycopg_driver(monkeypatch):
    monkeypatch.setattr(
        storage,
        "settings",
        replace(storage.settings, database_url="postgres://user:pass@localhost/verimind"),
    )
    storage.get_engine.cache_clear()

    try:
        url = storage._build_database_url()

        assert url.drivername == "postgresql+psycopg"
        assert make_url(str(url)).database == "verimind"
    finally:
        storage.get_engine.cache_clear()
