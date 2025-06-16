"""This test module contains all tests regarding the svarmiq approximator."""

from __future__ import annotations

import pytest

from shapiq.approximator.montecarlo import SVARMIQ
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize(
    ("n", "max_order", "index", "top_order"),
    [
        (7, 2, "SII", False),
        (7, 2, "SII", True),
        (7, 2, "k-SII", True),
        (7, 2, "k-SII", False),
        (7, 2, "STII", False),
        (7, 2, "STII", True),
        (7, 2, "FSII", True),
    ],
)
def test_initialization(n, max_order, index, top_order):
    """Tests the initialization of the SVARM-IQ approximator."""
    approximator = SVARMIQ(n, max_order, index=index, top_order=top_order)
    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is top_order
    assert approximator.min_order == (max_order if top_order else 0)
    assert approximator.iteration_cost == 1
    assert approximator.index == index


@pytest.mark.parametrize(
    ("n", "max_order", "top_order", "budget"),
    [(7, 2, False, 100), (7, 2, True, 100), (7, 2, False, 300)],
)
def test_approximate_sii(n, max_order, top_order, budget):
    """Tests the approximation of the SVARM-IQ SII approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approximator = SVARMIQ(n, max_order, index="SII", top_order=top_order, random_state=42)
    estimates = approximator.approximate(budget, game)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == max_order
    assert estimates.min_order == (max_order if top_order else 0)

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    # for order 2 (max_order) the interaction between player 1 and 2 is the most important (1.0)
    assert estimates[interaction] == pytest.approx(1.0, 0.01)

    if not top_order:
        # for order 1 (min_order) the interaction between  1 and 2 is the most important (0.6429)
        assert estimates[(1,)] == pytest.approx(0.6429, 0.05)
        assert estimates[(2,)] == pytest.approx(0.6429, 0.05)
