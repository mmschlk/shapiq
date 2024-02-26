"""This test module contains all tests regarding the SII regression approximator."""
from copy import deepcopy, copy

import numpy as np
import pytest

from interaction_values import InteractionValues
from approximator.regression._base import Regression
from approximator.regression import RegressionSII
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
    """Tests the initialization of the Regression approximator for SII."""
    approximator = RegressionSII(n, max_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is False
    assert approximator.min_order == 1
    assert approximator.iteration_cost == 1
    assert approximator.index == "SII"

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
    with pytest.raises(ValueError):
        _ = Regression(n, max_order, index="something")


@pytest.mark.parametrize(
    "n, max_order, budget, batch_size", [(7, 2, 380, 100), (7, 2, 380, None), (7, 2, 100, None)]
)
def test_approximate(n, max_order, budget, batch_size):
    """Tests the approximation of the Regression approximator for SII."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = RegressionSII(n, max_order, random_state=42)
    sii_estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(sii_estimates, InteractionValues)
    assert sii_estimates.max_order == max_order
    assert sii_estimates.min_order == 1

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # check that the estimates are correct
    # for order 1 player 1 and 2 are the most important with 0.6429
    assert sii_estimates[(1,)] == pytest.approx(0.6429, 0.4)  # quite a large interval
    assert sii_estimates[(2,)] == pytest.approx(0.6429, 0.4)

    # for order 2 the interaction between player 1 and 2 is the most important
    assert sii_estimates[(1, 2)] == pytest.approx(1.0, 0.2)

    # check efficiency
    efficiency = np.sum(sii_estimates.values[:n])
    assert efficiency == pytest.approx(2.0, 0.01)

    # try covert to nSII
    nsii_estimates = approximator.transforms_sii_to_ksii(sii_estimates)
    assert nsii_estimates.index == "k-SII"
