"""This test module contains all tests regarding the SV KernelSHAP regression approximator."""
from copy import deepcopy, copy

import numpy as np
import pytest

from approximator._base import InteractionValues
from approximator.regression import KernelSHAP
from games import DummyGame


@pytest.mark.parametrize(
    "n",
    [
        3,
        7,  # used in subsequent tests
        10,
    ],
)
def test_initialization(n):
    """Tests the initialization of the RegressionFSI approximator."""
    approximator = KernelSHAP(n)
    assert approximator.n == n
    assert approximator.max_order == 1
    assert approximator.top_order is False
    assert approximator.min_order == 1
    assert approximator.iteration_cost == 1
    assert approximator.index == "SV"

    approximator_copy = copy(approximator)
    approximator_deepcopy = deepcopy(approximator)
    approximator_deepcopy.index = "something"
    assert approximator_copy == approximator  # check that the copy is equal
    assert approximator_deepcopy != approximator  # check that the deepcopy is not equal
    approximator_string = str(approximator)
    assert repr(approximator) == approximator_string
    assert hash(approximator) == hash(approximator_copy)
    assert hash(approximator) != hash(approximator_deepcopy)
    with pytest.raises(ValueError):
        _ = approximator == 1


@pytest.mark.parametrize("n, budget, batch_size", [(7, 380, 100), (7, 380, None), (7, 100, None)])
def test_approximate(n, budget, batch_size):
    """Tests the approximation of the RegressionFSI approximator."""

    # TODO Change this to KernelSHAP
