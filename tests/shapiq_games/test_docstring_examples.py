"""The docstrings of shapiq_games: their examples run as written, and the API reference lists them."""

from __future__ import annotations

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
