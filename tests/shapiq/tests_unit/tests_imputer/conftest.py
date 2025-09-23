"""This module contains fixtures for imputer tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from typing import Any

import numpy as np
import pytest


@pytest.fixture(scope="package")
def dummy_model() -> Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Defines a simple placeholder model for testing."""

    def predict(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray(np.sum(x, axis=-1), dtype=float)

    return predict
