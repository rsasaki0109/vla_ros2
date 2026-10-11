"""Static checks for the Colab notebooks (no execution, no kernel)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def test_notebooks_exist() -> None:
    assert [path.name[:2] for path in NOTEBOOKS] == ["01", "02", "03", "04"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_structure_and_references(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    text = "\n".join("".join(cell["source"]) for cell in cells)
    assert f"blob/main/notebooks/{path.name}" in text, "Colab badge must point at this file"
    for cell in cells:
        assert cell.get("id"), "cells need ids (nbformat >= 4.5)"
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "commit notebooks without outputs"
            # IPython magics (! and %) are not Python; check the rest parses.
            source = "".join(cell["source"])
            python = "\n".join(
                line[: len(line) - len(line.lstrip())] + "pass"
                if line.lstrip().startswith(("!", "%"))
                else line
                for line in source.splitlines()
            )
            ast.parse(python)
    for script in re.findall(r"scripts/[\w./-]+\.py", text):
        assert (REPO_ROOT / script).is_file(), f"{path.name} references missing {script}"
