"""Numerical and target checks for KernelSHAP-IQ's shared regression solver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import null_space

from shapiq.approximator import InconsistentKernelSHAPIQ, KernelSHAPIQ
from shapiq.approximator.regression import base


@pytest.mark.parametrize("estimator_class", [KernelSHAPIQ, InconsistentKernelSHAPIQ])
@pytest.mark.parametrize("index", ["SII", "k-SII"])
def test_underdetermined_interaction_regression_minimum_norm(monkeypatch, estimator_class, index):
    """Unidentified directions must not acquire arbitrary large coefficients."""
    n = 8
    coefficients = np.arange(1, n + 1, dtype=float)
    original_solver = base.solve_regression
    solved_systems = []
    queried = []

    def record_solver(X, y, kernel_weights, *, use_svd=False):
        result = original_solver(X, y, kernel_weights, use_svd=use_svd)
        solved_systems.append((X.copy(), result.copy()))
        return result

    def game(coalitions):
        queried.extend(coalitions)
        return 7 + coalitions @ coefficients + 2 * np.prod(coalitions[:, :3], axis=1)

    monkeypatch.setattr(base, "solve_regression", record_solver)
    estimate = estimator_class(n=n, max_order=3, index=index, random_state=1).approximate(
        budget=32, game=game
    )

    assert len(queried) == 32
    assert estimate.baseline_value == 7
    assert np.all(np.isfinite(estimate.values))
    assert len(solved_systems) == (3 if estimator_class is KernelSHAPIQ else 1)
    underdetermined = False
    for design, solution in solved_systems:
        # Positive row weights preserve this null space. A minimum-norm solution
        # is orthogonal to every direction that leaves fitted values unchanged.
        unidentified_directions = null_space(design)
        if unidentified_directions.shape[1]:
            underdetermined = True
            np.testing.assert_allclose(unidentified_directions.T @ solution, 0, atol=1e-7, rtol=0)
    assert underdetermined


@pytest.mark.parametrize("estimator_class", [KernelSHAPIQ, InconsistentKernelSHAPIQ])
@pytest.mark.parametrize("index", ["SII", "k-SII"])
def test_full_budget_cubic_game_analytic_interactions(estimator_class, index):
    """A represented cubic has known SII and efficient order-three k-SII values."""
    n = 7
    coefficients = np.arange(1, n + 1, dtype=float)
    estimate = estimator_class(n=n, max_order=3, index=index, random_state=0).approximate(
        budget=2**n,
        game=lambda z: 7 + z @ coefficients + 2 * np.prod(z[:, :3], axis=1),
    )
    expected = []
    for interaction in estimate.dict_values:
        if not interaction:
            expected.append(7)
            continue
        value = coefficients[interaction[0]] if len(interaction) == 1 else 0
        if set(interaction).issubset({0, 1, 2}):
            if index == "SII":
                value += 2 / (4 - len(interaction))
            elif len(interaction) == 3:
                value += 2
        expected.append(value)

    # Finite endpoint penalties leave a small numerical reference residual.
    np.testing.assert_allclose(estimate.values, expected, rtol=0, atol=1e-7)
    if index == "k-SII":
        assert sum(estimate.values) == pytest.approx(7 + sum(coefficients) + 2, abs=1e-7)
    else:
        # Efficiency applies to first-order SII, not the sum of all SII orders.
        assert sum(estimate[(i,)] for i in range(n)) == pytest.approx(
            sum(coefficients) + 2, abs=1e-7
        )
