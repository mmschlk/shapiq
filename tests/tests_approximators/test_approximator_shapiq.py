"""This test module contains all tests regarding the shapiq approximator."""

from copy import copy, deepcopy

import numpy as np
import pytest

from shapiq.approximator.montecarlo import SHAPIQ
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize(
    "n, max_order, index, top_order",
    [
        (7, 2, "SII", False),
        (7, 2, "SII", True),
        (7, 2, "STII", False),
        (7, 2, "STII", True),
        (7, 2, "FSII", True),
    ],
)
def test_initialization(n, max_order, index, top_order):
    """Tests the initialization of the ShapIQ approximator."""

    approximator = SHAPIQ(n, max_order, index=index, top_order=top_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is top_order
    assert approximator.min_order == (max_order if top_order else 0)
    assert approximator.iteration_cost == 1
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


@pytest.mark.parametrize("n, max_order, budget", [(7, 2, 100), (7, 2, 100)])
def test_approximate_fsi(n, max_order, budget):
    """Tests the approximation of the ShapIQ FSII approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = SHAPIQ(n, max_order, index="FSII", top_order=True, random_state=42)
    estimates = approximator.approximate(budget, game)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == max_order  # only top order for FSII

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # for order 2 (max_order) the interaction between player 1 and 2 is the most important (1.0)
    interaction_estimate = estimates[interaction]
    assert interaction_estimate == pytest.approx(1.0, 0.4)  # large tolerance for FSII


@pytest.mark.parametrize(
    "n, max_order, top_order, budget",
    [(7, 2, False, 100), (7, 2, True, 100), (7, 2, False, 300)],
)
def test_approximate_sii(n, max_order, top_order, budget):
    """Tests the approximation of the ShapIQ SII approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = SHAPIQ(n, max_order, index="SII", top_order=top_order, random_state=42)
    estimates = approximator.approximate(budget, game)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if top_order else 0)

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # for order 2 (max_order) the interaction between player 1 and 2 is the most important (1.0)
    assert estimates[interaction] == pytest.approx(1.0, 0.4)

    if not top_order:
        # for order 1 (min_order) the interaction between  1 and 2 is the most important (0.6429)
        if budget <= 2**n:
            assert estimates[(1,)] == pytest.approx(0.6429, 0.4)
            assert estimates[(2,)] == pytest.approx(0.6429, 0.4)
        else:
            assert estimates[(1,)] == pytest.approx(0.6429, 0.01)
            assert estimates[(2,)] == pytest.approx(0.6429, 0.01)


@pytest.mark.parametrize("n, max_order, top_order, budget", [(7, 2, False, 100), (7, 2, True, 100)])
def test_approximate_sti(n, max_order, top_order, budget):
    """Tests the approximation of the ShapIQ STII approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = SHAPIQ(n, max_order, index="STII", top_order=top_order, random_state=42)
    estimates = approximator.approximate(budget, game)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if top_order else 0)

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # for order 2 (max_order) the interaction between player 1 and 2 is the most important (1.0)
    assert estimates[interaction] == pytest.approx(1.0, 0.5)

    if not top_order:
        # for order 1 (min_order) the interaction between player 1 and 2 is the most important (0.7)
        assert estimates[(1,)] == pytest.approx(1 / 7, 0.4)
        assert estimates[(2,)] == pytest.approx(1 / 7, 0.4)

        # check efficiency
        assert np.sum(estimates.values) == pytest.approx(2.0, 0.4)
