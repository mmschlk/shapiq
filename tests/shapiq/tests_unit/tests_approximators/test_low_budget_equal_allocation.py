"""Default low-budget equal allocation spends two queries; opt-out retains regression."""

from __future__ import annotations

from copy import deepcopy

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

    for budget in (2, 6, 12, 24, np.int64(36)):
        calls.clear()
        estimator = method(n)
        assert estimator.low_budget_equal_allocation is True
        estimate = estimator.approximate(budget, game)
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], [np.zeros(n, bool), np.ones(n, bool)])
        np.testing.assert_array_equal(estimate.values[1:], np.full(n, 91 / n))
        assert estimate[()] == estimate.baseline_value == 7
        assert estimate.estimation_budget == 2 and estimate.estimated
        assert estimate.index == "SV"
        assert sum(estimate[(i,)] for i in range(n)) == pytest.approx(91)


@pytest.mark.parametrize("method", [LeverageSHAP, OddSHAP])
@pytest.mark.parametrize(("n", "budget"), [(1, 2), (2, 4), (3, 8), (4, 13), (5, 32)])
def test_full_enumeration_and_above_three_n_keep_regression(method, n, budget):
    if method is OddSHAP and n == 1:
        for options in ({}, {"low_budget_equal_allocation": False}):
            with pytest.raises(ValueError, match="undefined for n <= 1"):
                method(n, **options)
        return

    def game(z):
        return 3 + z @ np.arange(1, n + 1) + np.prod(z[:, : min(n, 3)], axis=1)

    baseline = method(n, random_state=4, low_budget_equal_allocation=False).approximate(
        budget, game
    )
    default = method(n, random_state=4).approximate(budget, game)
    np.testing.assert_array_equal(default.values, baseline.values)
    assert default.baseline_value == baseline.baseline_value
    assert default.estimation_budget == baseline.estimation_budget
    assert default.estimated == baseline.estimated


@pytest.mark.parametrize("method", [LeverageSHAP, OddSHAP])
def test_three_n_boundary_skips_sampling_surrogate_and_rng(method, monkeypatch):
    n = 4
    estimator = method(n, random_state=17)
    rng_before = deepcopy(estimator._rng.bit_generator.state)
    sampler_before = deepcopy(estimator._sampler._rng.bit_generator.state)
    calls = []

    def game(z):
        calls.append(z.copy())
        return 3 + z @ np.arange(1, n + 1) + 2 * z[:, 0] * z[:, 1]

    def forbidden(*args, **kwargs):
        pytest.fail("Low-budget fallback must not sample, fit a surrogate or solve regression")

    with monkeypatch.context() as patch:
        if method is LeverageSHAP:
            patch.setattr(estimator, "_sample", forbidden)
        else:
            patch.setattr(estimator._sampler, "sample", forbidden)
            patch.setattr(estimator, "_fit_surrogate_model", forbidden)
        result = estimator.approximate(3 * n, game)
    np.testing.assert_array_equal(result.values, [3, 3, 3, 3, 3])
    assert result.estimation_budget == 2
    assert sum(len(z) for z in calls) == 2
    assert estimator._rng.bit_generator.state == rng_before
    assert estimator._sampler._rng.bit_generator.state == sampler_before
    # The subsequent 3n+1 call sees exactly the original sampling state.
    calls.clear()
    above = estimator.approximate(3 * n + 1, game)
    assert sum(len(z) for z in calls) > 2
    reference = method(n, random_state=17, low_budget_equal_allocation=False).approximate(
        3 * n + 1, game
    )
    np.testing.assert_array_equal(above.values, reference.values)
    assert above.estimation_budget == reference.estimation_budget


@pytest.mark.parametrize("method", [LeverageSHAP, OddSHAP])
def test_opt_out_uses_nonuniform_regression_with_original_query_budget(method):
    n, budget = 8, 16
    calls = []

    def game(z):
        calls.append(z.copy())
        return 7 + 3 * z[:, 0] + 2 * z[:, 0] * z[:, 1]

    result = method(n, random_state=0, low_budget_equal_allocation=False).approximate(budget, game)
    assert 2 < sum(len(z) for z in calls) <= budget
    assert result.estimation_budget == budget
    assert result.baseline_value == 7
    assert sum(result[(i,)] for i in range(n)) == pytest.approx(5)
    assert not np.allclose(result.values[1:], np.full(n, 5 / n))


@pytest.mark.parametrize("method", [LeverageSHAP, OddSHAP])
def test_fallback_rejects_invalid_input_without_sampling(method):
    estimator = method(5)
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
