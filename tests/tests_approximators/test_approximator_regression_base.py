"""This module contains all tests regarding the base Regression approximator"""

import pytest

from shapiq.approximator.regression import Regression
from shapiq.approximator.regression._base import AVAILABLE_INDICES_REGRESSION


def test_basic_functions():
    """Tests the initialization of the Regression approximator."""
    for index in AVAILABLE_INDICES_REGRESSION:
        _ = Regression(n=7, max_order=2, index=index)

    with pytest.raises(ValueError):
        _ = Regression(n=7, max_order=2, index="wrong_index")
