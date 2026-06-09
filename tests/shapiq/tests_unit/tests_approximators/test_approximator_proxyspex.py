"""Tests for the ProxySPEX approximator."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from shapiq.approximator.proxy.proxyspex import ProxySPEX
from shapiq.interaction_values import InteractionValues
from tests.shapiq.markers import skip_if_no_lightgbm


@skip_if_no_lightgbm
def test_initialization_defaults():
    """Test that ProxySPEX initializes with correct defaults."""
    n = 10
    proxyspex = ProxySPEX(n=n, max_order=2)

    # Check ProxySPEX default values
    assert proxyspex.n == n
    assert proxyspex.max_order == 2  # Default is 2
    assert proxyspex.index == "k-SII"
    assert proxyspex.top_order is False


@pytest.mark.parametrize(
    ("n", "index", "max_order", "top_order"),
    [
        (7, "STII", 2, False),
        (7, "FBII", 3, True),
        (20, "FSII", 5, False),
    ],
)
@skip_if_no_lightgbm
def test_initialization_custom(n, index, max_order, top_order):
    """Test ProxySPEX initialization with custom parameters."""
    proxyspex = ProxySPEX(
        n=n,
        index=index,
        max_order=max_order,
        top_order=top_order,
    )

    assert proxyspex.n == n
    assert proxyspex.max_order == max_order
    assert proxyspex.index == index
    assert proxyspex.top_order is top_order


@pytest.mark.parametrize(
    ("n", "interactions", "budget"),
    [
        (10, {(), (1,), (1, 2)}, 1000),
        (7, {(), (1,), (1, 2)}, 800),
    ],
)
@skip_if_no_lightgbm
def test_approximate(n, interactions, budget):
    """Test ProxySPEX approximation functionality."""

    def dummy_game(X):
        return [sum(1 for interaction in interactions if all(x[i] for i in interaction)) for x in X]

    # Initialize SPEX approximator
    proxyspex = ProxySPEX(n=n, max_order=n, random_state=42)

    # Perform approximation
    estimates = proxyspex.approximate(budget, dummy_game)

    # Verify the result structure
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == n
    assert estimates.min_order == 0  # Default top_order is False
    assert estimates.index == "k-SII"
    assert estimates.estimated
    assert estimates.estimation_budget > 0

    # Check that values are not empty
    assert len(estimates.values) > 0

    # Check that the target interaction has a non-zero value
    for interaction in interactions:
        assert interaction in estimates.interaction_lookup
        assert abs(estimates[interaction]) > 0
        # The dummy game should return approximately 1.0 for the target interaction
        assert estimates[interaction] == pytest.approx(1.0, abs=0.5)


def test_proxyspex_falls_back_to_decision_tree_without_lightgbm(monkeypatch):
    """Without lightgbm, ProxySPEX falls back to a bare DecisionTreeRegressor instead of raising.

    ProxySPEX no longer requires lightgbm: the default ``"lightgbm"`` tag warns and falls back
    to a scikit-learn ``DecisionTreeRegressor`` (so it is *not* wrapped in the LightGBM grid
    search), keeping ProxySPEX usable without the optional gradient-boosting backends.
    """
    from sklearn.tree import DecisionTreeRegressor

    # Temporarily pretend lightgbm is not installed
    monkeypatch.setitem(sys.modules, "lightgbm", None)

    with pytest.warns(UserWarning):
        approximator = ProxySPEX(n=10, max_order=2)
    assert isinstance(approximator.proxy_model, DecisionTreeRegressor)


@skip_if_no_lightgbm
def test_refine_zero_total_energy():
    """Tests that ``_refine`` handles the case when total Fourier energy is zero.

    When all non-baseline Fourier coefficients are zero the method must return the input
    dictionary unchanged instead of attempting to divide by zero.
    """
    n = 5
    approximator = ProxySPEX(n=n, index="FBII", max_order=2, random_state=42)

    four_dict = {
        (): 1.0,  # baseline coefficient
        (0,): 0.0,
        (1,): 0.0,
        (0, 1): 0.0,
        (2,): 0.0,
        (1, 2): 0.0,
    }
    train_X = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
        ]
    )
    train_y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    result = approximator._refine(four_dict, train_X, train_y)

    assert result == four_dict
    assert len(result) == len(four_dict)
