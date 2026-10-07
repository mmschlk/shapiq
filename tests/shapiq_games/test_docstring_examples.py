"""The examples in the docstrings of shapiq_games and shapiq_benchmark run as written."""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import shapiq_benchmark
import shapiq_games
from tests.shapiq_games.helpers import is_installed


def _modules(package: object) -> list[str]:
    prefix = f"{package.__name__}."  # type: ignore[attr-defined]
    names = [name for _, name, _ in pkgutil.walk_packages(package.__path__, prefix)]  # type: ignore[attr-defined]
    return sorted([package.__name__, *names])  # type: ignore[attr-defined]


# examples that need an optional package to run
_REQUIRES = {"shapiq_games.vision.image_classifier": "skimage"}


@pytest.mark.parametrize("name", _modules(shapiq_games) + _modules(shapiq_benchmark))
def test_docstring_examples_run(name: str) -> None:
    required = _REQUIRES.get(name)
    if required is not None and not is_installed(required):
        pytest.skip(f"{required} is not installed")
    module = importlib.import_module(name)
    result = doctest.testmod(
        module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE, report=True
    )
    assert result.failed == 0, f"{result.failed} docstring example(s) failed in {name}"
