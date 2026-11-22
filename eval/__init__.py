"""Evaluation package compatibility facade for the legacy evaluator module."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_LEGACY_PATH = Path(__file__).resolve().parents[1] / "eval.py"
_SPEC = importlib.util.spec_from_file_location("verimind_legacy_eval", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load evaluator at {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LEGACY)

LABELS = _LEGACY.LABELS
evaluation_mode = _LEGACY.evaluation_mode
local_verdict = _LEGACY.local_verdict
normalize_wiki_title = _LEGACY.normalize_wiki_title
read_cases = _LEGACY.read_cases
retrieval_hit = _LEGACY.retrieval_hit
summarize = _LEGACY.summarize
wilson_interval = _LEGACY.wilson_interval
