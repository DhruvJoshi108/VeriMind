"""Simple demo runner for the verification pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verimind.pipeline import VerificationPipeline


def main() -> None:
    pipeline = VerificationPipeline()
    sample = "Paris is the capital of France. The Eiffel Tower is in Paris."
    report = pipeline.verify_text(sample)
    print(report)


if __name__ == "__main__":
    main()
