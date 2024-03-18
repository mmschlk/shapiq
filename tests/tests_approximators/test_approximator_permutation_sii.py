"""This test module contains all tests regarding the SII permutation sampling approximator."""
from copy import copy, deepcopy

import numpy as np
import pytest

from interaction_values import InteractionValues
from approximator.permutation import PermutationSamplingSII
from games import DummyGame


@pytest.mark.parametrize(
    "n, max_order, top_order, index, expected",
    [
        (3, 1, True, "SII", 6),
        (3, 1, False, "SII", 6),
        (3, 2, True, "SII", 8),
        (3, 2, False, "SII", 14),
        (10, 3, False, "SII", 120),
        (10, 3, False, "k-SII", 120),
        (10, 3, False, "something", 120),  # expected to fail with ValueError
    ],
)
def test_initialization(n, max_order, top_order, index, expected):
    """Tests the initialization of the PermutationSamplingSII approximator."""
    if index == "something":
        with pytest.raises(ValueError):
            _ = PermutationSamplingSII(n, max_order, index, top_order)
        return
    approximator = PermutationSamplingSII(n, max_order, index, top_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order == top_order
    assert approximator.min_order == (max_order if top_order else 1)
    assert approximator.iteration_cost == expected
    assert approximator.index == index

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
    "n, max_order, top_order, budget, batch_size",
    [
        (7, 2, False, 380, 10),
        (7, 2, False, 500, 10),
        (7, 2, False, 500, None),
        (7, 2, True, 500, None),
    ],
)
def test_approximate(n, max_order, top_order, budget, batch_size):
    """Tests the approximation of the PermutationSamplingSII approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = PermutationSamplingSII(n, max_order, "SII", top_order, random_state=42)
    sii_estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(sii_estimates, InteractionValues)
    assert sii_estimates.max_order == max_order
    assert sii_estimates.min_order == (max_order if top_order else 1)

    # check that the budget is respected
    assert game.access_counter <= budget

    # check that the estimates are correct
    if not top_order:
        # for order 1 player 1 and 2 are the most important with 0.6429
        assert sii_estimates[(1,)] == pytest.approx(0.6429, 0.4)  # quite a large interval
        assert sii_estimates[(2,)] == pytest.approx(0.6429, 0.4)

    # for order 2 the interaction between player 1 and 2 is the most important
    assert sii_estimates[(1, 2)] == pytest.approx(1.0, 0.2)

    # check efficiency
    if not top_order:
        efficiency = np.sum(sii_estimates.values[:n])
        assert efficiency == pytest.approx(2.0, 0.01)
