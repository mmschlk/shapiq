"""This test module contains all tests regarding the kADDSHAP regression approximator."""

from __future__ import annotations

import pytest

from shapiq.approximator import kADDSHAP
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame


@pytest.mark.parametrize("n", [3, 7, 10])
def test_initialization(n):
    """Tests the initialization of the kADDSHAP approximator."""
    approximator = kADDSHAP(n)
    assert approximator.n == n
    assert approximator.max_order == 2
    assert approximator.index == "kADD-SHAP"
    assert approximator.top_order is False
    assert approximator.min_order == 0
    assert approximator.iteration_cost == 1


@pytest.mark.parametrize(("budget", "order"), [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_approximate(budget, order):
    """Tests the approximation of the kADDSHAP approximator."""
    n = 7
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    approximator = kADDSHAP(n, max_order=order)
    sii_estimates = approximator.approximate(budget, game)
    assert isinstance(sii_estimates, InteractionValues)
    assert sii_estimates.max_order == order
    assert sii_estimates.min_order == 0

    # check that the budget is respected
    assert game.access_counter <= budget

    assert sii_estimates[(1,)] == pytest.approx(0.6429, abs=0.1)
    assert sii_estimates[(2,)] == pytest.approx(0.6429, abs=0.1)


def test_initialization_with_sv_index():
    """Tests that the kADDSHAP approximator can be set up to estimate Shapley values."""
    approximator = kADDSHAP(7, index="SV")
    assert approximator.index == "SV"
    assert approximator.approximation_index == "kADD-SHAP"
    assert approximator.max_order == 2

    with pytest.raises(ValueError, match="Invalid index"):
        kADDSHAP(7, index="SII")


@pytest.mark.parametrize("order", [2, 3])
def test_approximate_sv(order):
    """Tests that with index="SV" the order-1 part of the k-additive solution is returned."""
    n = 7
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    approximator = kADDSHAP(n, max_order=order, index="SV")
    sv_estimates = approximator.approximate(100, game)
    assert isinstance(sv_estimates, InteractionValues)
    assert sv_estimates.index == "SV"
    assert sv_estimates.max_order == 1
    assert sv_estimates.min_order == 0

    assert sv_estimates[(1,)] == pytest.approx(0.6429, abs=0.1)
    assert sv_estimates[(2,)] == pytest.approx(0.6429, abs=0.1)
    assert sv_estimates[(0,)] == pytest.approx(0.1429, abs=0.1)
