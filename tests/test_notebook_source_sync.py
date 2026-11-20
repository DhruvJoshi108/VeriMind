import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK_DIRS = ("api", "dashboard", "retrieval", "scripts", "tests", "verimind")


def test_generated_notebooks_have_matching_python_sources():
    for directory in SOURCE_NOTEBOOK_DIRS:
        for notebook_path in (PROJECT_ROOT / directory).glob("*.ipynb"):
            notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
            markdown = "".join(notebook["cells"][0].get("source", []))
            if "generated directly from the corresponding Python source file" not in markdown:
                continue

            match = re.match(r"^# `([^`]+\.py)`", markdown)
            assert match, f"Notebook source path is missing: {notebook_path}"

            source_path = PROJECT_ROOT / match.group(1)
            code_cells = [
                cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
            ]
            assert source_path.is_file(), f"Missing Python source: {source_path}"
            assert len(code_cells) == 1, f"Unexpected source-cell count: {notebook_path}"
            assert source_path.read_text(encoding="utf-8") == "".join(
                code_cells[0].get("source", [])
            ), f"Python source is out of sync: {source_path}"
