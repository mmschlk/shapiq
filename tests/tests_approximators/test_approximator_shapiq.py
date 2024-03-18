"""This test module contains all tests regarding the shapiq approximator."""

from copy import copy, deepcopy

import numpy as np
import pytest
from approximator.shapiq import ShapIQ
from games import DummyGame
from interaction_values import InteractionValues


@pytest.mark.parametrize(
    "n, max_order, index, top_order",
    [
        (7, 2, "SII", False),
        (7, 2, "SII", True),
        (7, 2, "STI", False),
        (7, 2, "STI", True),
        (7, 2, "FSI", True),
        (7, 2, "FSI", False),  # expected to fail
    ],
)
def test_initialization(n, max_order, index, top_order):
    """Tests the initialization of the ShapIQ approximator."""
    try:
        approximator = ShapIQ(n, max_order, index=index, top_order=top_order)
    except ValueError:
        if index == "FSI" and not top_order:
            return
        raise
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is top_order
    assert approximator.min_order == (max_order if top_order else 1)
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
    with pytest.raises(ValueError):
        approximator_deepcopy._weight_kernel(3, 2)


@pytest.mark.parametrize("n, max_order, budget, batch_size", [(7, 2, 100, None), (7, 2, 100, 10)])
def test_approximate_fsi(n, max_order, budget, batch_size):
    """Tests the approximation of the ShapIQ FSI approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = ShapIQ(n, max_order, index="FSI", top_order=True, random_state=42)
    estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == max_order  # only top order for FSI

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # for order 2 (max_order) the interaction between player 1 and 2 is the most important (1.0)
    interaction_estimate = estimates[interaction]
    assert interaction_estimate == pytest.approx(1.0, 0.4)  # large tolerance for FSI


@pytest.mark.parametrize(
    "n, max_order, top_order, budget, batch_size",
    [(7, 2, False, 100, None), (7, 2, True, 100, 10), (7, 2, False, 300, None)],
)
def test_approximate_sii(n, max_order, top_order, budget, batch_size):
    """Tests the approximation of the ShapIQ SII approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = ShapIQ(n, max_order, index="SII", top_order=top_order, random_state=42)
    estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if top_order else 1)

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
            assert estimates[(1,)] == pytest.approx(0.6429, 0.0001)
            assert estimates[(2,)] == pytest.approx(0.6429, 0.0001)

        # check efficiency
        assert np.sum(estimates.values[:n]) == pytest.approx(2.0, 0.4)


@pytest.mark.parametrize(
    "n, max_order, top_order, budget, batch_size", [(7, 2, False, 100, None), (7, 2, True, 100, 10)]
)
def test_approximate_sti(n, max_order, top_order, budget, batch_size):
    """Tests the approximation of the ShapIQ STI approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = ShapIQ(n, max_order, index="STI", top_order=top_order, random_state=42)
    estimates = approximator.approximate(budget, game, batch_size=batch_size)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if top_order else 1)

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
