"""Tests for the ProxySPEX approximator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from shapiq.approximator.proxy import ProxySHAP
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues


def test_initialization_defaults():
    """Test that ProxySHAP initializes with correct defaults."""
    n = 10
    proxyshap = ProxySHAP(n=n)

    # Check ProxySHAP default values
    assert proxyshap.n == n
    assert proxyshap.max_order == 2
    assert proxyshap.index == "k-SII"
    assert isinstance(proxyshap.proxy_model, XGBRegressor)


@pytest.mark.parametrize(
    ("n", "index", "max_order"),
    [
        (7, "STII", 2),
        (7, "FBII", 3),
        (20, "FSII", 20),
    ],
)
def test_initialization_custom(n, index, max_order):
    """Test ProxySHAP initialization with custom parameters."""
    proxyshap = ProxySHAP(
        n=n,
        index=index,
        max_order=max_order,
    )

    assert proxyshap.n == n
    assert proxyshap.max_order == (n if max_order is None else max_order)
    assert proxyshap.index == index


@pytest.mark.parametrize(
    ("n", "interactions", "budget"),
    [
        (10, {(), (1,), (1, 2)}, 1024),
        (7, {(), (1,), (1, 2)}, 128),
    ],
)
def test_approximate(n, interactions, budget):
    """Test ProxySHAP approximation functionality."""

    def dummy_game(X):
        return np.array(
            [sum(1 for interaction in interactions if all(x[i] for i in interaction)) for x in X]
        )

    # Initialize ProxySHAP approximator
    proxyshap = ProxySHAP(n=n, random_state=42, index="k-SII", max_order=2)

    exact_computer = ExactComputer(game=dummy_game, n_players=n)
    gt_values = exact_computer(index="k-SII", order=2)
    # Perform approximation
    estimates = proxyshap.approximate(budget, dummy_game)

    # Verify the result structure
    assert isinstance(estimates, InteractionValues)
    assert estimates.max_order == 2
    assert estimates.min_order == 0  # Default top_order is False
    assert estimates.index == "k-SII"
    # estimated follows the codebase convention: exact only at/above full enumeration.
    assert estimates.estimated == (budget < 2**n)
    assert estimates.estimation_budget > 0

    # Check that values are not empty
    assert len(estimates.values) > 0

    for interaction in interactions:
        if interaction == ():
            continue
        assert np.allclose(estimates[interaction], gt_values[interaction], atol=1e-5)


@pytest.mark.parametrize("adjustment", ["none", "msr"])
@pytest.mark.parametrize(
    ("n", "max_order", "index", "interactions"),
    [
        # main effects only -> the order-1 (degree-1) linear proxy represents the game exactly
        (6, 1, "SV", {(), (0,), (2,), (4,)}),
        # up to pairwise -> the order-2 interaction-only expansion represents the game exactly
        (6, 2, "k-SII", {(), (1,), (1, 2), (0, 3)}),
    ],
)
def test_linear_proxy_recovers_exact_interactions(n, max_order, index, interactions, adjustment):
    """A ``"linear"`` proxy recovers a multilinear game exactly via its polynomial expansion.

    The dummy game is itself a degree-``max_order`` interaction-only polynomial of the binary
    coalitions, so the linear proxy -- fit on the matching ``PolynomialFeatures`` expansion -- fits
    it exactly at full budget and its coefficients are the game's Moebius coefficients. The
    extracted interactions must therefore match :class:`ExactComputer` to machine precision, for
    both the unadjusted and MSR-adjusted routes (the residual is ~0, so the adjustment is a no-op).
    """

    def game(X):
        return np.array(
            [sum(1 for interaction in interactions if all(x[i] for i in interaction)) for x in X]
        )

    proxyshap = ProxySHAP(
        n=n,
        max_order=max_order,
        index=index,
        proxy_model="linear",
        adjustment=adjustment,
        random_state=0,
    )
    # the "linear" tag resolves to a bare scikit-learn LinearRegression (the linear route)
    assert isinstance(proxyshap.proxy_model, LinearRegression)

    gt_values = ExactComputer(game=game, n_players=n)(index=index, order=max_order)
    estimates = proxyshap.approximate(2**n, game)

    assert isinstance(estimates, InteractionValues)
    assert estimates.index == index
    assert estimates.max_order == max_order
    for interaction in interactions:
        if interaction == ():
            continue
        assert np.allclose(estimates[interaction], gt_values[interaction], atol=1e-6)


def test_hpo_flag_wraps_xgboost_proxy_in_default_grid():
    """``hpo=True`` wraps a string-resolved boosting proxy in its default GridSearchCV.

    The default (``hpo=False``) keeps the resolved ``"xgboost"`` proxy bare; ``hpo=True`` wraps it
    in a :class:`~sklearn.model_selection.GridSearchCV` over the shared ``_XGBOOST_DECODER_GRID``.
    """
    from sklearn.model_selection import GridSearchCV

    from shapiq.approximator.proxy._models import _XGBOOST_DECODER_GRID

    bare = ProxySHAP(n=6, max_order=2)  # default hpo=False
    assert isinstance(bare.proxy_model, XGBRegressor)

    tuned = ProxySHAP(n=6, max_order=2, hpo=True)  # default proxy_model="xgboost"
    assert isinstance(tuned.proxy_model, GridSearchCV)
    assert isinstance(tuned.proxy_model.estimator, XGBRegressor)
    assert tuned.proxy_model.param_grid == _XGBOOST_DECODER_GRID

    # the linear tag is never HPO-wrapped, even with hpo=True
    linear = ProxySHAP(n=6, max_order=2, proxy_model="linear", hpo=True)
    assert isinstance(linear.proxy_model, LinearRegression)
