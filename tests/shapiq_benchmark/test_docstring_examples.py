"""The docstrings of shapiq_benchmark: their examples run as written, and the API reference lists them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import shapiq_benchmark
from tests.shapiq_games.helpers import (
    package_modules,
    run_docstring_examples,
    undocumented_names,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("name", package_modules(shapiq_benchmark))
def test_docstring_examples_run(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the examples cache ground truth; keep it out of the user's cache, which would also hide
    # the computation from later runs
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    run_docstring_examples(name)


def test_the_api_reference_lists_every_public_name() -> None:
    """A new setup (or other name in an ``__all__``) also needs a line in the API reference."""
    assert undocumented_names(shapiq_benchmark, "api_reference_benchmark.rst") == []
