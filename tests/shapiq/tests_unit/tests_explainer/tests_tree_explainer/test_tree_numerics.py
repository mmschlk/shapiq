"""Tests for the guarded Chebyshev-Vandermonde solves in ``shapiq.tree._numerics``."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebpts2

from shapiq.tree._numerics import grid_is_certified, solve_vandermonde
from shapiq.tree.linear.explainer import LinearTreeSHAP, get_N_v2


@pytest.mark.parametrize("grid_size", [4, 8, 12, 20, 26])
def test_matches_explicit_inverse_on_well_conditioned_grids(grid_size):
    """The fast path agrees with the previous ``inv(V) @ rhs`` formulation."""
    D = chebpts2(grid_size)
    rng = np.random.default_rng(0)
    for i in range(2, grid_size + 1):
        rhs = rng.normal(size=i)
        V = np.vander(D[:i]).T
        expected = np.linalg.inv(V).dot(rhs)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning may fire on the fast path
            result = solve_vandermonde(D[:i], rhs, certified=grid_is_certified(D))
        np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-10)


def test_precision_warning_fires_at_interior_prefixes_of_large_grids():
    """Conditioning peaks at interior prefixes (i ~ n/2): for a size-30 grid the
    warning must fire even though the degenerate prefix is shorter than the safe
    grid size — this is the regression test for the mis-calibrated size gate."""
    D = chebpts2(28)
    i = 14  # peak conditioning region; full-rank with cond ~4e12 (comfortably inside
    # the precision-warning band: rank-deficient prefixes only appear from grid ~32)
    with pytest.warns(RuntimeWarning, match="condition number"):
        solve_vandermonde(D[:i], np.ones(i))


def test_no_warning_for_ordinary_grids():
    D = chebpts2(12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i in range(2, 13):
            solve_vandermonde(D[:i], np.ones(i))


def test_singular_system_returns_least_squares_fallback_with_warning():
    """A rank-deficient system must not crash; it returns a least-squares solution
    and warns that the values are not reliable."""
    points = np.concatenate([chebpts2(30), chebpts2(30)[:30]])  # duplicated nodes
    size = len(points)
    with pytest.warns(RuntimeWarning, match="NOT reliable"):
        result = solve_vandermonde(points, np.ones(size), certified=grid_is_certified(points))
    assert result.shape == (size,)
    assert np.all(np.isfinite(result))


def test_deep_tree_constructs_and_explains_with_warning():
    """End-to-end: LinearTreeSHAP on a deep tree no longer raises LinAlgError."""
    sklearn = pytest.importorskip("sklearn.tree")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 10))
    y = rng.normal(size=20000)
    tree = sklearn.DecisionTreeRegressor(max_depth=60, min_samples_leaf=1, random_state=0).fit(X, y)
    if tree.get_depth() < 50:
        pytest.skip("random data did not produce a deep enough tree")
    with pytest.warns(RuntimeWarning):
        explainer = LinearTreeSHAP(tree)
    values = explainer.explain_function(X[0])
    assert np.all(np.isfinite(values.values))


def test_one_summary_warning_per_matrix_construction():
    """A deep grid touches many degenerate prefixes; the construction loop must
    emit exactly one coalesced warning, not one per prefix."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_N_v2(chebpts2(30))
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1


def test_custom_clustered_grid_is_not_silently_fast_pathed():
    """Certification measures the actual grid: a small but badly conditioned custom
    grid (legal via LinearTreeSHAP's base_func) must not take the unchecked path."""
    grid = np.linspace(0.0, 1.0, 26) ** 8  # clustered near zero, size <= 26
    assert not grid_is_certified(grid)
    assert grid_is_certified(chebpts2(26))
    with pytest.warns(RuntimeWarning):
        for i in range(2, len(grid) + 1):
            solve_vandermonde(grid[:i], np.ones(i), certified=grid_is_certified(grid))
