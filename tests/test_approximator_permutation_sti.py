"""This test module contains all tests regarding the STI permutation sampling approximator."""
import numpy as np
import pytest

from approximator._base import InteractionValues
from approximator.permutation import PermutationSamplingSTI
from games import DummyGame


@pytest.mark.parametrize(
    "n, max_order, iteration_cost",
    [
        (3, 1, 6),
        (3, 1, 6),
        (3, 2, 12),
        (3, 2, 12),
        (7, 2, 84),  # used in subsequent tests
        (10, 3, 960),
    ],
)
def test_initialization(n, max_order, iteration_cost):
    """Tests the initialization of the PermutationSamplingSTI approximator."""
    approximator = PermutationSamplingSTI(n, max_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is False
    assert approximator.min_order == 1
    assert approximator.iteration_cost == iteration_cost
    assert approximator.index == "STI"


@pytest.mark.parametrize(
    "n, max_order, budget, batch_size",
    [
        (7, 2, 380, 2),
        (7, 2, 500, 2),
    ],
)
def test_approximate(n, max_order, budget, batch_size):
    """Tests the approximation of the PermutationSamplingSTI approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = PermutationSamplingSTI(n, max_order, random_state=42)
    sti_estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(sti_estimates, InteractionValues)
    assert sti_estimates.max_order == max_order
    assert sti_estimates.min_order == 1

    # check that the budget is respected
    assert game.access_counter <= budget

    # check that the estimates are correct
    # for order 1 all players should be equal
    first_order_values = np.asarray([sti_estimates[(i,)] for i in range(n)])
    assert np.allclose(first_order_values, first_order_values[0])
    # for order 2 the interaction between player 1 and 2 is the most important (1.0)
    assert sti_estimates[(1, 2)] == pytest.approx(1.0, 0.2)

    # check efficiency
    efficiency = np.sum(sti_estimates.values)
    assert efficiency == pytest.approx(2.0, 0.01)