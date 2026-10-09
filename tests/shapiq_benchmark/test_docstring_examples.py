"""The examples in the docstrings of shapiq_benchmark run as written."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import shapiq_benchmark
from tests.shapiq_games.helpers import package_modules, run_docstring_examples

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("name", package_modules(shapiq_benchmark))
def test_docstring_examples_run(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the examples cache ground truth; keep it out of the user's cache, which would also hide
    # the computation from later runs
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    run_docstring_examples(name)
