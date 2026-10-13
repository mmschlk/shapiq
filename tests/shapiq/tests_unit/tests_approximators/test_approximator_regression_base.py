"""This module contains all tests regarding the base Regression approximator."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from shapiq.approximator.regression import (
    KernelSHAP,
    Regression,
    RegressionFBII,
    RegressionFSII,
    kADDSHAP,
)
from shapiq.approximator.regression.base import ValidRegressionIndices, solve_regression


def test_basic_functions():
    """Tests the initialization of the Regression approximator."""
    for index in set(get_args(ValidRegressionIndices)):
        _ = Regression(n=7, max_order=2, index=index)

    with pytest.raises(ValueError):
        _ = Regression(n=7, max_order=2, index="wrong_index")


@pytest.mark.parametrize("use_svd", [False, True])
def test_solve_regression_full_rank(use_svd):
    """Well-conditioned weighted least squares agrees with the normal equations."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    y = rng.standard_normal(20)
    weights = np.linspace(0.5, 2.0, 20)
    expected = np.linalg.solve(X.T @ (weights[:, None] * X), X.T @ (weights * y))

    result = solve_regression(X=X, y=y, kernel_weights=weights, use_svd=use_svd)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_solve_regression_dependent_columns():
    """Duplicate columns share their coefficient equally in the minimum-norm solution."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 4))
    X[:, 2] = X[:, 1]
    y = X @ np.array([1.0, 2.0, 4.0, -1.5])
    weights = np.geomspace(1.0, 1e8, 20)

    result = solve_regression(X=X, y=y, kernel_weights=weights)
    np.testing.assert_allclose(result, [1.0, 3.0, 3.0, -1.5], rtol=1e-10, atol=1e-10)


def test_solve_regression_large_finite_weights():
    """Finite weights must not overflow an unnecessary Gram-matrix construction."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = np.array([2.0, -3.0])
    result = solve_regression(X=X, y=X @ expected, kernel_weights=np.full(3, 1e308))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_solve_regression_underdetermined_minimum_norm():
    """A finite solution and a small residual do not establish minimum norm."""
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, size=(12, 24)).astype(float)
    y = rng.standard_normal(12)
    # X has full row rank; the small, well-conditioned dual system is an
    # independent reference for its unique minimum-norm interpolating solution.
    expected = X.T @ np.linalg.solve(X @ X.T, y)
    result = solve_regression(X=X, y=y, kernel_weights=np.ones(12))
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("estimator_class", "kwargs", "budget_factor"),
    [
        (KernelSHAP, {}, 1),
        (RegressionFSII, {"max_order": 1}, 1),
        (kADDSHAP, {"max_order": 1}, 1),
        (RegressionFBII, {"max_order": 2}, 2),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_low_budget_additive_game_minimum_norm(estimator_class, kwargs, budget_factor, seed):
    """A feasible exact coefficient vector bounds the minimum-norm solution."""
    n = 11
    coefficients = np.arange(1, n + 1, dtype=float)
    estimator = estimator_class(n=n, random_state=seed, **kwargs)
    estimate = estimator.approximate(budget=budget_factor * n, game=lambda z: z @ coefficients)

    # All higher-order and empty coefficients are zero in this additive game.
    # Low budgets need not recover each value, but exploding norms are incorrect.
    assert np.linalg.norm(estimate.values) <= np.linalg.norm(coefficients) * (1 + 1e-8)


@pytest.mark.parametrize("estimator_class", [KernelSHAP, RegressionFSII, RegressionFBII, kADDSHAP])
def test_full_budget_additive_game(estimator_class):
    """With all coalitions available, every method recovers the additive values."""
    n = 7
    coefficients = np.arange(1, n + 1, dtype=float)
    kwargs = {} if estimator_class is KernelSHAP else {"max_order": 2}
    estimate = estimator_class(n=n, random_state=0, **kwargs).approximate(
        budget=2**n, game=lambda z: z @ coefficients
    )
    expected = [coefficients[key[0]] if len(key) == 1 else 0.0 for key in estimate.dict_values]
    np.testing.assert_allclose(estimate.values, expected, rtol=1e-8, atol=1e-8)
