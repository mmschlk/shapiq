"""This test module contains all tests regarding the SV KernelSHAP regression approximator."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.approximator.regression import KernelSHAP
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize("n", [3, 7, 10])
def test_initialization(n):
    """Tests the initialization of the RegressionFSII approximator."""
    approximator = KernelSHAP(n)
    assert approximator.n == n
    assert approximator.max_order == 1
    assert approximator.top_order is False
    assert approximator.min_order == 0
    assert approximator.iteration_cost == 1


@pytest.mark.parametrize(("n", "budget"), [(7, 380), (7, 100)])
def test_approximate(n, budget):
    """Tests the approximation of the KernelSHAP approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    approximator = KernelSHAP(n)
    sv_estimates = approximator.approximate(budget, game)
    assert isinstance(sv_estimates, InteractionValues)
    assert sv_estimates.max_order == 1
    assert sv_estimates.min_order == 0
    assert sv_estimates.index == "SV"
    assert sv_estimates.estimation_budget <= budget
    assert sv_estimates.estimated != (budget >= 2**n)

    # check that the budget is respected
    assert game.access_counter <= budget

    # check that the values are in the correct range
    # check that the estimates are correct
    # for order 1 player 1 and 2 are the most important with 0.6429
    assert sv_estimates[(1,)] == pytest.approx(0.6429, abs=0.1)
    assert sv_estimates[(2,)] == pytest.approx(0.6429, abs=0.1)

    # check efficiency
    efficiency = np.sum(sv_estimates.values)
    assert efficiency == pytest.approx(2.0, abs=0.1)
