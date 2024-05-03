"""This test module contains all tests regarding the kADDSHAP regression approximator."""

import copy

import pytest

from shapiq.games.benchmark import DummyGame
from shapiq.approximator import kADDSHAP
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize("n", [3, 7, 10])
def test_initialization(n):
    """Tests the initialization of the kADDSHAP approximator."""
    approximator = kADDSHAP(n)
    assert approximator.n == n
    assert approximator.max_order == 1
    assert approximator.index == "kADD-SHAP"
    assert approximator.top_order is False
    assert approximator.min_order == 0
    assert approximator.iteration_cost == 1

    approximator_copy = copy.copy(approximator)
    approximator_deepcopy = copy.deepcopy(approximator)
    approximator_deepcopy.index = "something"
    assert approximator_copy == approximator  # check that the copy is equal
    assert approximator_deepcopy != approximator  # check that the deepcopy is not equal
    approximator_string = str(approximator)
    assert repr(approximator) == approximator_string
    assert hash(approximator) == hash(approximator_copy)
    assert hash(approximator) != hash(approximator_deepcopy)
    with pytest.raises(ValueError):
        _ = approximator == 1


@pytest.mark.parametrize("budget, order", [(100, 1), (100, 2), (100, 3), (100, 4)])
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

    if order > 1:
        assert sii_estimates[(1, 2)] == pytest.approx(1.0, abs=0.1)

    assert sii_estimates[(1, 2, 3)] == pytest.approx(0, abs=0.1)
