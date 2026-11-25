"""Dashboard-only configuration; browser secrets stay in Streamlit session state."""

from __future__ import annotations

import math
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


def _positive_float(name: str, raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _positive_int(name: str, raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
    batch_limit_name = "VERIMIND_DASHBOARD_BATCH_LIMIT"
    batch_limit_value = os.getenv(batch_limit_name)
    if batch_limit_value is None:
        batch_limit_name = "MAX_CLAIMS_PER_DOC"
        batch_limit_value = os.getenv(batch_limit_name, "50")
    return DashboardConfig(
        api_url=os.getenv("VERIMIND_API_URL", "http://localhost:8000").rstrip("/"),
        api_key=os.getenv("VERIMIND_API_KEY", os.getenv("API_KEY", "")),
        timeout_seconds=_positive_float(
            "VERIMIND_DASHBOARD_TIMEOUT_SECONDS",
            os.getenv("VERIMIND_DASHBOARD_TIMEOUT_SECONDS", "90"),
        ),
        max_characters=20_000,
        batch_limit=_positive_int(batch_limit_name, batch_limit_value),
        eval_results_dir=Path(os.getenv("VERIMIND_EVAL_RESULTS_DIR", root / "eval" / "results")),
    )
