"""This test module contains all tests regarding the SVARM approximator."""

from __future__ import annotations

import pytest

from shapiq.approximator.montecarlo import SVARM
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize("n", [7, 10, 50])
def test_initialization(n):
    """Tests the initialization of the SVARM approximator."""
    approximator = SVARM(n=n)
    assert approximator.n == n
    assert approximator.max_order == 1
    assert approximator.iteration_cost == 1
    assert approximator.index == "SV"
    assert approximator.min_order == 0


@pytest.mark.parametrize("n, budget", [(7, 100), (7, 300)])
def test_approximate_sii(n, budget):
    """Tests the approximation of the SVARM SV approximation."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    svarm = SVARM(n, random_state=42)
    estimates = svarm.approximate(budget, game)
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == 1

    # check that the budget is respected
    assert game.access_counter <= budget + 2

    assert estimates[(1,)] == pytest.approx(0.6429, 0.05)
    assert estimates[(2,)] == pytest.approx(0.6429, 0.05)
