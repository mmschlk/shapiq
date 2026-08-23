"""This module contains all tests regarding the base Regression approximator."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from shapiq.approximator.regression import Regression
from shapiq.approximator.regression.base import ValidRegressionIndices, solve_regression


def test_basic_functions():
    """Tests the initialization of the Regression approximator."""
    for index in set(get_args(ValidRegressionIndices)):
        _ = Regression(n=7, max_order=2, index=index)

    with pytest.raises(ValueError):
        _ = Regression(n=7, max_order=2, index="wrong_index")


def test_solve_regression_full_rank_fast_path():
    """use_svd=False on a well-conditioned full-rank system should use np.linalg.solve and return the correct solution."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = 20, 4
    X = rng.standard_normal((n_rows, n_cols))
    true_coef = np.array([1.0, -2.0, 3.0, -1.5])
    y = X @ true_coef
    weights = np.ones(n_rows)

    result = solve_regression(X=X, y=y, kernel_weights=weights, use_svd=False)
    np.testing.assert_allclose(result, true_coef, atol=1e-8)


def test_solve_regression_rank_deficient_falls_back_to_lstsq():
    """use_svd=False with a rank-deficient Gram matrix must fall back to lstsq without crashing.

    When two columns of X are identical the Gram matrix is singular (rank < n_cols),
    so np.linalg.solve would raise LinAlgError. The guard must detect this and fall
    back to lstsq, returning a finite result.
    """
    rng = np.random.default_rng(1)
    n_rows, n_cols = 20, 4
    X = rng.standard_normal((n_rows, n_cols))
    X[:, 2] = X[:, 1]  # make column 2 identical to column 1 → rank-deficient Gram matrix
    y = rng.standard_normal(n_rows)
    weights = np.ones(n_rows)

    result = solve_regression(X=X, y=y, kernel_weights=weights, use_svd=False)
    assert result.shape == (n_cols,)
    assert np.all(np.isfinite(result)), f"Expected finite result, got {result}"


def test_solve_regression_gram_nan_guard():
    """use_svd=False must return all-NaN (not crash) when the Gram matrix contains Inf/NaN.

    Extreme kernel weights cause Inf in WX and therefore in X^T @ WX. The guard must
    catch this before passing it to np.linalg.solve, which would silently return NaNs
    or raise depending on the platform.
    """
    rng = np.random.default_rng(2)
    n_rows, n_cols = 10, 3
    X = rng.standard_normal((n_rows, n_cols))
    y = rng.standard_normal(n_rows)
    weights = np.ones(n_rows)
    weights[0] = np.inf  # causes Inf in WX → Inf in Gram matrix

    result = solve_regression(X=X, y=y, kernel_weights=weights, use_svd=False)
    assert result.shape == (n_cols,)
    assert np.all(np.isnan(result)), f"Expected all-NaN, got {result}"


def test_solve_regression_underdetermined_no_svd():
    """use_svd=False with fewer rows than columns (underdetermined) falls back to lstsq.

    The rank check (X.shape[0] < X.shape[1]) triggers lstsq, which returns the
    minimum-norm solution. The result must be finite.
    """
    rng = np.random.default_rng(3)
    n_rows, n_cols = 3, 6  # underdetermined: more unknowns than equations
    X = rng.standard_normal((n_rows, n_cols))
    y = rng.standard_normal(n_rows)
    weights = np.ones(n_rows)

    result = solve_regression(X=X, y=y, kernel_weights=weights, use_svd=False)
    assert result.shape == (n_cols,)
    assert np.all(np.isfinite(result)), f"Expected finite result, got {result}"
