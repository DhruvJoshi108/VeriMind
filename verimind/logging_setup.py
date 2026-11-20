"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any


def build_logger(name: str = "verimind") -> logging.Logger:
    """Create a JSON-like logger for local debugging and diagnostics."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **details: Any) -> None:
    """Emit a structured event with a trace id if present."""
    payload = {"event": event, **details}
    logger.info(json.dumps(payload, default=str, ensure_ascii=False))


def make_trace_id() -> str:
    """Generate a request-scoped trace id."""
    return str(uuid.uuid4())
