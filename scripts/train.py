"""Train the model on a synthetic claim-evidence dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verimind.data import build_synthetic_dataset
from verimind.nli_model import NLIModel


def main() -> None:
    dataset = build_synthetic_dataset()
    model = NLIModel()
    model.fit(dataset["pairs"], dataset["labels"])
    model.save(Path("artifacts/nli_model.pkl"))
    print("trained")


if __name__ == "__main__":
    main()
