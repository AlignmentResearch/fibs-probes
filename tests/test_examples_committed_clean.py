"""The committed example notebooks hold no outputs.

An executed notebook carries the text of what it printed, which makes its diffs noisy and can carry data or local
paths into the repository. Run a local copy instead (the example README shows how; ``*.local.ipynb`` is gitignored).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = subprocess.run(
    ["git", "ls-files", "--", "*.ipynb"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.split()


def test_there_are_notebooks_to_check() -> None:
    assert NOTEBOOKS


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook_has_no_outputs(notebook: str) -> None:
    content = json.loads((ROOT / notebook).read_text())
    cells = content["cells"]
    assert any(cell["cell_type"] == "code" for cell in cells)
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
        assert not cell.get("attachments")
    assert "widgets" not in content.get("metadata", {})
