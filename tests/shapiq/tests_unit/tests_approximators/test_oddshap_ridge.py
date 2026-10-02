"""Optional ridge preserves OddSHAP constraints and its unregularized default."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import OddSHAP


@pytest.mark.parametrize("ridge", [-1, np.nan, np.inf])
def test_invalid_penalty(ridge):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        OddSHAP(4, ridge=ridge)


def test_ridge_matches_independent_constrained_solution():
    estimator = OddSHAP(4, ridge=0.001)
    estimator.n_active_interactions = 4
    X = np.array([[1, 1, 1.00001], [2, 2.00001, 2], [3, 3.00001, 3.00001]])
    y = np.array([1.0, -1.0, 2.0])
    result = estimator._solve_constrained_regression(
        X_tilde=X, y_tilde=y, empty_set_value=2.0, full_set_value=8.0, ridge=0.001
    )
    # Independent KKT solve for min ||X beta-y||² + ridge ||beta||²,
    # constrained to sum(beta)=-(8-2)/2. This penalty differs from penalizing
    # the free coordinates only by a constant on the feasible hyperplane.
    KKT = np.block(
        [[X.T @ X + 0.001 * np.eye(3), np.ones((3, 1))], [np.ones((1, 3)), np.zeros((1, 1))]]
    )
    expected = np.linalg.solve(KKT, np.r_[X.T @ y, -3.0])[:3]
    np.testing.assert_allclose(result[1:], expected, atol=1e-10)
    assert result[0] == 5.0
    assert result[1:].sum() == pytest.approx(-3.0)


@pytest.mark.parametrize("budget", [12, 13, 16])
def test_budget_gate_and_default(budget, monkeypatch):
    n = 4

    def game(z):
        return 2 + z @ np.arange(1, n + 1) + 5 * np.prod(z[:, :3], axis=1)

    regularized = OddSHAP(n, ridge=0.001, random_state=0)
    calls = []
    original = regularized._solve_constrained_regression

    def capture(**kwargs):
        calls.append(kwargs["ridge"])
        return original(**kwargs)

    monkeypatch.setattr(regularized, "_solve_constrained_regression", capture)
    estimate = regularized.approximate(budget, game)
    assert calls == [0.001 if budget <= 3 * n and budget < 2**n else 0.0]
    assert sum(estimate[(i,)] for i in range(n)) == pytest.approx(15.0)
    baseline = OddSHAP(n, random_state=0).approximate(budget, game)
    explicit = OddSHAP(n, ridge=0.0, random_state=0).approximate(budget, game)
    np.testing.assert_array_equal(baseline.values, explicit.values)
    if budget > 3 * n or budget == 2**n:
        np.testing.assert_array_equal(estimate.values, baseline.values)


def test_one_term_and_no_interior_rows():
    estimator = OddSHAP(4)
    estimator.n_active_interactions = 2
    for X, y in [(np.ones((3, 1)), np.ones(3)), (np.empty((0, 1)), np.empty(0))]:
        result = estimator._solve_constrained_regression(
            X_tilde=X, y_tilde=y, empty_set_value=2.0, full_set_value=8.0, ridge=0.001
        )
        np.testing.assert_array_equal(result, [5.0, -3.0])


def test_full_enumeration_within_low_budget_window(monkeypatch):
    estimator = OddSHAP(3, ridge=0.001, random_state=0)
    original = estimator._solve_constrained_regression
    penalties = []

    def capture(**kwargs):
        penalties.append(kwargs["ridge"])
        return original(**kwargs)

    monkeypatch.setattr(estimator, "_solve_constrained_regression", capture)
    estimator.approximate(8, lambda z: z.sum(axis=1))
    assert penalties == [0.0]  # 2**3 is still below 3 * 3.
