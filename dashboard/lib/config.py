"""Dashboard-only configuration; browser secrets stay in Streamlit session state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class DashboardConfig:
    api_url: str
    api_key: str
    timeout_seconds: float
    max_characters: int
    batch_limit: int
    eval_results_dir: Path


def display_api_origin(url: str) -> str:
    """Return only the non-secret origin portion of a configured URL."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "Configured API endpoint"
    if parsed.scheme not in {"http", "https"} or not hostname:
        return "Configured API endpoint"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    if port is not None:
        hostname = f"{hostname}:{port}"
    return f"{parsed.scheme}://{hostname}"


def load_config() -> DashboardConfig:
    root = Path(__file__).resolve().parents[2]
    return DashboardConfig(
        api_url=os.getenv("VERIMIND_API_URL", "http://localhost:8000").rstrip("/"),
        api_key=os.getenv("VERIMIND_API_KEY", os.getenv("API_KEY", "")),
        timeout_seconds=float(os.getenv("VERIMIND_DASHBOARD_TIMEOUT_SECONDS", "90")),
        max_characters=20_000,
        batch_limit=int(
            os.getenv("VERIMIND_DASHBOARD_BATCH_LIMIT", os.getenv("MAX_CLAIMS_PER_DOC", "50"))
        ),
        eval_results_dir=Path(os.getenv("VERIMIND_EVAL_RESULTS_DIR", root / "eval" / "results")),
    )
