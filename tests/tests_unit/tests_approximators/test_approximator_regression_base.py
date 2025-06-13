"""This module contains all tests regarding the base Regression approximator."""

from __future__ import annotations

from typing import get_args

import pytest

from shapiq.approximator.regression import Regression
from shapiq.approximator.regression._base import ValidRegressionIndices


def test_basic_functions():
    """Tests the initialization of the Regression approximator."""
    for index in set(get_args(ValidRegressionIndices)):
        _ = Regression(n=7, max_order=2, index=index)

    with pytest.raises(ValueError):
        _ = Regression(n=7, max_order=2, index="wrong_index")
