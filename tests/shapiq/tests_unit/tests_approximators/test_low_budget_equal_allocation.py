"""The opt-in endpoint baseline preserves defaults and spends exactly two queries."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import LeverageSHAP, OddSHAP
from shapiq.game_theory.exact import ExactComputer


@pytest.mark.parametrize("method", [LeverageSHAP, OddSHAP])
def test_low_budget_equal_allocation_and_query_accounting(method):
    n = 12
    calls = []

    def game(z):
        calls.append(z.copy())
        return 7 + z @ np.arange(1, n + 1) + 13 * np.prod(z[:, :3], axis=1)

    for budget in (6, 12, 24, np.int64(36)):
        calls.clear()
        estimate = method(n, low_budget_equal_allocation=True).approximate(budget, game)
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], [np.zeros(n, bool), np.ones(n, bool)])
        np.testing.assert_array_equal(estimate.values[1:], np.full(n, 91 / n))
        assert estimate[()] == estimate.baseline_value == 7
        assert estimate.estimation_budget == 2 and estimate.estimated
        assert estimate.index == "SV"
        assert sum(estimate[(i,)] for i in range(n)) == pytest.approx(91)


@pytest.mark.parametrize("method", [LeverageSHAP, OddSHAP])
def test_defaults_and_outside_fallback_match_existing_estimator(method):
    for n, budget in ((1, 2), (3, 8), (4, 13), (5, 5)):
        if method is OddSHAP and n == 1:
            for enabled in (False, True):
                with pytest.raises(ValueError, match="undefined for n <= 1"):
                    method(n, low_budget_equal_allocation=enabled)
            continue

        def game(z, n=n):
            return 3 + z @ np.arange(1, n + 1) + np.prod(z[:, : min(n, 3)], axis=1)

        baseline = method(n, random_state=4).approximate(budget, game)
        disabled = method(n, random_state=4, low_budget_equal_allocation=False).approximate(
            budget, game
        )
        np.testing.assert_array_equal(disabled.values, baseline.values)
        assert disabled.estimation_budget == baseline.estimation_budget
        if budget > 3 * n or budget >= 2**n:
            enabled = method(n, random_state=4, low_budget_equal_allocation=True).approximate(
                budget, game
            )
            np.testing.assert_array_equal(enabled.values, baseline.values)
            assert enabled.estimation_budget == baseline.estimation_budget
            assert enabled.estimated == baseline.estimated


@pytest.mark.parametrize("method", [LeverageSHAP, OddSHAP])
def test_fallback_rejects_invalid_input_without_sampling(method):
    estimator = method(5, low_budget_equal_allocation=True)
    calls = []

    def game(z):
        calls.append(z)
        return z.sum(axis=1)

    for budget in (-1, 0, 1, 2.5, np.nan, np.inf, True):
        with pytest.raises(ValueError, match="integer of at least two"):
            estimator.approximate(budget, game)
    assert calls == []
    for endpoints in ([0, np.nan], [0, np.inf], [[0], [1]], [0], 0):
        with pytest.raises(ValueError, match="two finite scalar endpoint"):
            estimator.approximate(5, lambda _, endpoints=endpoints: endpoints)
    with pytest.raises(TypeError, match="must be a bool"):
        method(5, low_budget_equal_allocation=1)


def test_equal_baseline_nmse_bound_for_nonlinear_games():
    """Efficiency makes the equal vector the projection onto the constant direction."""
    n = 5
    rng = np.random.default_rng(4)
    for _ in range(4):
        values = rng.normal(size=2**n)

        def game(z, values=values):
            return values[z.astype(int) @ (1 << np.arange(n))]

        truth = ExactComputer(game, n_players=n)("SV", 1)
        phi = np.array([truth[(i,)] for i in range(n)])
        for method in (LeverageSHAP, OddSHAP):
            estimate = method(n, low_budget_equal_allocation=True).approximate(10, game)
            pred = np.array([estimate[(i,)] for i in range(n)])
            assert np.sum((pred - phi) ** 2) / np.sum(phi**2) <= 1 + 1e-14
            assert pred.sum() == pytest.approx(values[-1] - values[0])


@pytest.mark.parametrize("method", [LeverageSHAP, OddSHAP])
def test_endpoint_difference_handles_subnormal_and_overflow(method):
    """Finite representable equal shares survive both endpoint magnitude extremes."""
    estimator = method(2, low_budget_equal_allocation=True)
    for magnitude in (np.nextafter(0.0, 1.0), np.finfo(float).max):
        with np.errstate(over="raise", invalid="raise"):
            estimate = estimator.approximate(
                2, lambda _, magnitude=magnitude: np.array([-magnitude, magnitude])
            )
        np.testing.assert_array_equal(estimate.values[1:], [magnitude, magnitude])
        assert estimate.baseline_value == -magnitude
        assert estimate.estimation_budget == 2
