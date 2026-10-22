"""The docstrings of shapiq_games: their examples run as written, and the API reference lists them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import shapiq_games
from tests.shapiq_games.helpers import (
    is_installed,
    package_modules,
    run_docstring_examples,
    undocumented_names,
)

# examples that need an optional package to run
_REQUIRES = {"shapiq_games.vision.image_classifier": "skimage"}


@pytest.mark.parametrize("name", package_modules(shapiq_games))
def test_docstring_examples_run(name: str) -> None:
    required = _REQUIRES.get(name)
    if required is not None and not is_installed(required):
        pytest.skip(f"{required} is not installed")
    run_docstring_examples(name)


def test_the_api_reference_lists_every_public_name() -> None:
    """A new game (or other name in an ``__all__``) also needs a line in the API reference."""
    assert undocumented_names(shapiq_games, "api_reference_games.rst") == []


def test_the_docs_cite_with_the_cite_role() -> None:
    """Every citation uses ``:cite:``, so the bibliography page lists every cited work.

    Footnote citations (``:footcite:``) never reach the bibliography page.
    """
    root = Path(__file__).resolve().parents[2]
    sources = [
        *(root / "src").rglob("*.py"),
        *(root / "examples").rglob("*.py"),
        *(root / "examples").rglob("*.rst"),
        *(root / "docs" / "source" / "_data").glob("*.toml"),
        *(root / "docs" / "source").glob("*.rst"),
        *(root / "docs" / "source" / "introduction").glob("*"),
    ]
    footnote_citations = re.compile(r":footcite|footbibliography")
    offenders = [
        str(path.relative_to(root))
        for path in sources
        if footnote_citations.search(path.read_text())
    ]
    assert offenders == []
