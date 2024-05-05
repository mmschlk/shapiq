"""This test module contains all tests regarding the SII permutation sampling approximator."""

from copy import copy, deepcopy

import numpy as np
import pytest

from shapiq.approximator.permutation import PermutationSamplingSII
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


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
    assert approximator.min_order == (max_order if top_order else 0)
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


@pytest.mark.parametrize("index", ["SII", "k-SII"])
@pytest.mark.parametrize(
    "n, max_order, top_order, budget, batch_size",
    [
        (7, 2, False, 380, 10),
        (7, 2, False, 500, 10),
        (7, 2, False, 500, None),
        (7, 2, True, 500, None),
    ],
)
def test_approximate(n, max_order, top_order, budget, batch_size, index):
    """Tests the approximation of the PermutationSamplingSII approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = PermutationSamplingSII(n, max_order, index, top_order, random_state=42)
    estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if (top_order and index is not "k-SII") else 0)
    assert estimates.index == index
    assert estimates.estimated is True  # always estimated
    assert estimates.estimation_budget <= budget

    # check that the budget is respected
    assert game.access_counter <= budget

    # check that the estimates are correct
    if not top_order:
        assert estimates[(0,)] == pytest.approx(0.1442, abs=0.2)

        if index == "SII":
            assert estimates[(1,)] == pytest.approx(0.6429, abs=0.2)  # large interval
            assert estimates[(2,)] == pytest.approx(0.6429, abs=0.2)
        if index == "k-SII":
            assert estimates[(1,)] == pytest.approx(0.1442, abs=0.2)
            assert estimates[(2,)] == pytest.approx(0.1442, abs=0.2)

    # for order 2 the interaction between player 1 and 2 is the most important
    assert estimates[(1, 2)] == pytest.approx(1.0, 0.2)
