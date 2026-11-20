"""Run the offline evaluator and save a timestamped JSON artifact under eval/results."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval import _LEGACY


def main() -> int:
    """Run the maintained evaluator, then copy its JSONL records into one JSON artifact."""
    before = set(_LEGACY.RESULTS_DIR.glob("eval_*.jsonl"))
    exit_code = _LEGACY.main()
    created = sorted(set(_LEGACY.RESULTS_DIR.glob("eval_*.jsonl")) - before)
    if not created:
        return exit_code
    records = [json.loads(line) for line in created[-1].read_text(encoding="utf-8").splitlines()]
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = output_dir / f"evaluation_{timestamp}.json"
    output.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"package_result_file": str(output), "legacy_result_file": str(created[-1])}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
