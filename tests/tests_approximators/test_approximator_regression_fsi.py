"""This test module contains all tests regarding the FSI regression approximator."""
from copy import deepcopy, copy

import numpy as np
import pytest

from interaction_values import InteractionValues
from approximator.regression import RegressionFSI
from games import DummyGame


@pytest.mark.parametrize(
    "n, max_order",
    [
        (3, 1),
        (3, 1),
        (3, 2),
        (3, 2),
        (7, 2),  # used in subsequent tests
        (10, 3),
    ],
)
def test_initialization(n, max_order):
    """Tests the initialization of the RegressionFSI approximator."""
    approximator = RegressionFSI(n, max_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is False
    assert approximator.min_order == 1
    assert approximator.iteration_cost == 1
    assert approximator.index == "FSI"

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


@pytest.mark.parametrize(
    "n, max_order, budget, batch_size", [(7, 2, 380, 100), (7, 2, 380, None), (7, 2, 100, None)]
)
def test_approximate(n, max_order, budget, batch_size):
    """Tests the approximation of the RegressionFSI approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = RegressionFSI(n, max_order, random_state=42)
    fsi_estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(fsi_estimates, InteractionValues)
    assert fsi_estimates.max_order == max_order
    assert fsi_estimates.min_order == 1

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # check that the estimates are correct

    # for order 1 all players should be equal
    first_order: np.ndarray = fsi_estimates.values[:n]  # fist n values are first order
    assert np.allclose(first_order, first_order[0])

    # for order 2 the interaction between player 1 and 2 is the most important (1.0)
    interaction_estimate = fsi_estimates[interaction]
    assert interaction_estimate == pytest.approx(1.0, 0.01)

    # check efficiency
    efficiency = np.sum(fsi_estimates.values)
    assert efficiency == pytest.approx(efficiency, 0.01)
