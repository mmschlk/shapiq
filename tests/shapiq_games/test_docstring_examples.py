"""The examples in the docstrings of shapiq_games run as written."""

from __future__ import annotations

import pytest

import shapiq_games
from tests.shapiq_games.helpers import is_installed, package_modules, run_docstring_examples

# examples that need an optional package to run
_REQUIRES = {"shapiq_games.vision.image_classifier": "skimage"}


@pytest.mark.parametrize("name", package_modules(shapiq_games))
def test_docstring_examples_run(name: str) -> None:
    required = _REQUIRES.get(name)
    if required is not None and not is_installed(required):
        pytest.skip(f"{required} is not installed")
    run_docstring_examples(name)
