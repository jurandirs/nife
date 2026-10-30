"""The documentation runs: the code of the README and every notebook of notebooks/.

They are run from the repository; from an installed package without them, the tests
are skipped.
"""

import contextlib
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NOTEBOOKS = sorted((ROOT / "notebooks").rglob("*.ipynb")) if (ROOT / "notebooks").is_dir() else []


@pytest.mark.skipif(not README.is_file(), reason="the README is not here")
def test_readme(tmp_path, monkeypatch):
    """The Python blocks of the README run, in order, and the output it shows is the one they print."""
    text = README.read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # what the examples write stays out of the repository
    printed = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(printed):
        for number, block in enumerate(re.findall(r"```python\n(.*?)```", text, re.S), 1):
            exec(compile(block, f"README.md, block {number}", "exec"), namespace)
    for shown in re.findall(r"```text\n(.*?)```", text, re.S):
        assert shown.strip() in printed.getvalue(), f"the README shows an output that its code does not print:\n{shown}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_notebook(path, tmp_path):
    """The notebook runs from top to bottom without errors."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()  # raises CellExecutionError, with the cell and the traceback, if a cell fails
