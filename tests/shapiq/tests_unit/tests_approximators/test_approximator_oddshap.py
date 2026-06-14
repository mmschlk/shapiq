"""Tests for the OddSHAP approximator.

OddSHAP estimates first-order Shapley values via paired sampling + odd-only
Fourier regression. A LightGBM surrogate is fitted to the sampled coalitions
and its exact Fourier spectrum (extracted via ProxySPEX's tree-to-Fourier
routine) screens the top ``ceil(budget / interaction_factor)`` odd
higher-order interactions; the active support is then solved with a
constrained weighted Fourier regression and the odd coefficients are
transformed into Shapley values. Budgets below ``n * interaction_factor``
are rejected with a ``ValueError`` (there is deliberately no fallback to
another estimator).

The constraint system enforces the exact identities
    beta_empty = (f(N) + f(empty)) / 2,
    sum over non-empty odd Fourier coefficients = -(f(N) - f(empty)) / 2,
which after the -2 scaling in `_transform_to_shapley` guarantees the
efficiency axiom by construction:
    sum_i phi_i = f(N) - f(empty),
    phi_empty   = f(empty).

These two identities are checked as EXACT properties below, alongside a
full-budget convergence check against ExactComputer on a sparse SOUM game.
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.special import binom

import shapiq
import shapiq.approximator as approximator_module
from shapiq.approximator.regression import OddSHAP
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import SOUM, DummyGame
from tests.shapiq.markers import skip_if_no_lightgbm

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_init_defaults(n):
    approx = OddSHAP(n=n)
    assert approx.n == n
    assert approx.max_order == 1
    assert approx.min_order == 0
    assert approx.top_order is False
    assert approx.index == "SV"
    assert approx.interaction_factor == 10


def test_init_custom_kwargs():
    custom_weights = np.zeros(9, dtype=float)
    custom_weights[1:8] = 1.0 / 7  # n=8, valid weights summing to 1
    approx = OddSHAP(
        n=8,
        interaction_factor=5,
        sampling_weights=custom_weights,
        random_state=123,
        tree_params={"max_depth": 4},
    )
    assert approx.interaction_factor == 5
    assert approx.tree_params == {"max_depth": 4}


@pytest.mark.parametrize("bad_eta", [0, -1])
def test_init_rejects_nonpositive_interaction_factor(bad_eta):
    """interaction_factor must be positive (guards a later ceil(budget/eta) division)."""
    with pytest.raises(ValueError, match="interaction_factor"):
        OddSHAP(n=8, interaction_factor=bad_eta)


def test_init_warns_without_lightgbm(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.warns(UserWarning, match="LightGBM is not installed"):
        approx = OddSHAP(n=8)
    from sklearn.tree import DecisionTreeRegressor

    assert isinstance(approx._surrogate_template, DecisionTreeRegressor)


def test_public_approximator_exports_include_oddshap():
    assert "OddSHAP" in approximator_module.__all__
    assert OddSHAP in approximator_module.SV_APPROXIMATORS


def test_top_level_export_includes_oddshap():
    """OddSHAP is re-exported at the package top level like its sibling approximators."""
    assert shapiq.OddSHAP is OddSHAP
    assert "OddSHAP" in shapiq.__all__


# -----------------------------------------------------------------------------
# Coalition-size sampling weights
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_sampling_weights_shape_and_sum(n):
    w = OddSHAP._init_sampling_weights_static(n)
    assert w.shape == (n + 1,)
    assert w.dtype == float
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_sampling_weights_zero_at_boundaries(n):
    w = OddSHAP._init_sampling_weights_static(n)
    assert w[0] == 0.0
    assert w[n] == 0.0


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_sampling_weights_symmetric(n):
    w = OddSHAP._init_sampling_weights_static(n)
    for k in range(1, n):
        assert w[k] == pytest.approx(w[n - k]), (
            f"weights not symmetric at k={k}: w[{k}]={w[k]} vs w[{n - k}]={w[n - k]}"
        )


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_sampling_weights_uniform_over_non_boundary_sizes(n):
    """Sampling weights are uniform over non-boundary coalition sizes.

    The paper's `1/((n-1)·C(n-2,k-1))` formula is the *regression kernel*
    weight (now in `_init_regression_kernel_weights_static`), not the
    sampling distribution. The coalition sampler uses uniform sampling.
    """
    w = OddSHAP._init_sampling_weights_static(n)
    # All non-boundary sizes share the same weight.
    expected_non_boundary = 1.0 / (n - 1)
    for k in range(1, n):
        assert w[k] == pytest.approx(expected_non_boundary)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_regression_kernel_weights_match_paper_formula(n):
    """Regression LSQ kernel weights follow the paper's `1/((n-1)·C(n-2,k-1))` up
    to a global scale — equivalent to the KernelSHAP weighting scheme.
    """
    w = OddSHAP._init_regression_kernel_weights_static(n)
    expected = np.zeros(n + 1, dtype=float)
    for k in range(1, n):
        expected[k] = 1.0 / ((n - 1) * binom(n - 2, k - 1))
    # Both arrays should be proportional with zero boundaries.
    assert w[0] == 0.0
    assert w[n] == 0.0
    ratios = w[1:n] / expected[1:n]
    np.testing.assert_allclose(ratios, ratios[0])  # constant ratio


# -----------------------------------------------------------------------------
# `approximate` return-value contract
# -----------------------------------------------------------------------------


def _regression_path_setup(n=8, seed=42):
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=seed)
    approx = OddSHAP(n=n, random_state=seed)
    budget = 2**n  # well above n*interaction_factor (= 10n) for n>=6
    assert budget >= n * approx.interaction_factor
    return approx, game, budget


def test_approximate_returns_interaction_values():
    approx, game, budget = _regression_path_setup()
    iv = approx.approximate(budget, game)
    assert isinstance(iv, InteractionValues)


def test_approximate_iv_field_contract():
    approx, game, budget = _regression_path_setup(n=8)
    iv = approx.approximate(budget, game)
    assert iv.index == "SV"
    assert iv.n_players == 8
    assert iv.max_order == 1
    assert iv.min_order == 0
    assert iv.values.shape == (9,)
    assert iv.values.dtype == float


def test_approximate_baseline_equals_empty_set_value():
    approx, game, budget = _regression_path_setup(n=8)
    iv = approx.approximate(budget, game)
    empty = float(game(np.zeros((1, 8), dtype=bool))[0])
    assert iv.baseline_value == pytest.approx(empty)
    assert iv.values[0] == pytest.approx(empty)


def test_approximate_estimation_budget_recorded():
    approx, game, budget = _regression_path_setup(n=8)
    iv = approx.approximate(budget, game)
    assert iv.estimation_budget == budget


@pytest.mark.parametrize("budget_pct", [0.5, 1.0])
def test_approximate_estimated_flag(budget_pct):
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    budget = int(budget_pct * 2**n)
    iv = approx.approximate(budget, game)
    assert iv.estimated == (budget < 2**n)


# -----------------------------------------------------------------------------
# Constraint-system identities (Theorem 3.2 + Appendix C)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 8, 10])
def test_efficiency_axiom_holds_exactly(n):
    """sum_i phi_i = v(N) - v(empty) is enforced exactly by the constraint system."""
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    iv = approx.approximate(2**n, game)
    v_full = float(game(np.ones((1, n), dtype=bool))[0])
    v_empty = float(game(np.zeros((1, n), dtype=bool))[0])
    assert np.sum(iv.values[1:]) == pytest.approx(v_full - v_empty, abs=1e-6)


@pytest.mark.parametrize("n", [6, 8, 10])
def test_baseline_exact(n):
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    iv = approx.approximate(2**n, game)
    v_empty = float(game(np.zeros((1, n), dtype=bool))[0])
    assert iv.values[0] == pytest.approx(v_empty, abs=1e-9)


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_determinism_same_seed_same_output():
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    a = OddSHAP(n=n, random_state=7).approximate(2**n, game)
    b = OddSHAP(n=n, random_state=7).approximate(2**n, game)
    np.testing.assert_array_equal(a.values, b.values)


def test_different_seed_differs():
    """Sub-budget regime — different seeds should sample different coalitions
    and therefore produce different Shapley estimates.
    """
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    budget = 120  # above n*interaction_factor=80, below 2^8=256
    a = OddSHAP(n=n, random_state=0).approximate(budget, game)
    b = OddSHAP(n=n, random_state=1).approximate(budget, game)
    assert not np.array_equal(a.values, b.values)


# -----------------------------------------------------------------------------
# Budget validation
#
# Sub-budget calls raise instead of silently falling back to TreeSHAP (a
# deliberate divergence from Algorithm 1); verify the threshold is enforced.
# -----------------------------------------------------------------------------


def test_low_budget_raises_value_error():
    """budget < n * interaction_factor must raise ValueError, not silently fall back."""
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    budget = n * approx.interaction_factor - 1
    with pytest.raises(ValueError, match="too small"):
        approx.approximate(budget, game)


def test_full_enumeration_bypasses_minimum_budget():
    """budget >= 2**n is accepted even when 2**n < n * interaction_factor (small n)."""
    n = 4  # 2**4 = 16 < 40 = n * eta
    game = DummyGame(n=n, interaction=(1, 2))
    iv = OddSHAP(n=n, random_state=0).approximate(2**n, game)
    assert isinstance(iv, InteractionValues)
    v_full = float(game(np.ones((1, n), dtype=bool))[0])
    v_empty = float(game(np.zeros((1, n), dtype=bool))[0])
    assert np.sum(iv.values[1:]) == pytest.approx(v_full - v_empty, abs=1e-6)


def test_small_budget_below_full_enumeration_still_raises():
    """Below both the eta-based minimum and 2**n the ValueError is kept, and the
    message reports the effective minimum (16 = 2**4, not n * eta = 40)."""
    n = 4
    game = DummyGame(n=n, interaction=(1, 2))
    with pytest.raises(ValueError, match="at least 16 evaluations"):
        OddSHAP(n=n, random_state=0).approximate(2**n - 1, game)


def test_boundary_budget_uses_paper_candidate_count(monkeypatch):
    """At budget = n * interaction_factor (the minimum permitted),
    `_select_odd_interactions` should be called with the paper's
    candidate count `ceil(budget / interaction_factor)`.
    """
    n = 8
    approx = OddSHAP(n=n, random_state=0)
    budget = n * approx.interaction_factor
    captured = {}

    def additive_game(coalitions):
        return coalitions.astype(float).sum(axis=1)

    def fake_fit_surrogate_model(*, coalitions, game_values):
        return object()

    def fake_select_odd_interactions(**kwargs):
        captured["n_candidate_interactions"] = kwargs["n_candidate_interactions"]
        return []

    monkeypatch.setattr(approx, "_fit_surrogate_model", fake_fit_surrogate_model)
    monkeypatch.setattr(approx, "_select_odd_interactions", fake_select_odd_interactions)

    approx.approximate(budget, additive_game)

    assert captured["n_candidate_interactions"] == max(
        0, math.ceil(budget / approx.interaction_factor) - n
    )


def test_candidate_interaction_count_matches_paper(monkeypatch):
    n = 8
    approx = OddSHAP(n=n, random_state=0)
    budget = n * approx.interaction_factor + 1
    captured = {}

    def additive_game(coalitions):
        return coalitions.astype(float).sum(axis=1)

    def fake_fit_surrogate_model(*, coalitions, game_values):
        return object()

    def fake_select_odd_interactions(**kwargs):
        captured["n_candidate_interactions"] = kwargs["n_candidate_interactions"]
        return []

    monkeypatch.setattr(approx, "_fit_surrogate_model", fake_fit_surrogate_model)
    monkeypatch.setattr(approx, "_select_odd_interactions", fake_select_odd_interactions)

    approx.approximate(budget, additive_game)

    assert captured["n_candidate_interactions"] == max(
        0, math.ceil(budget / approx.interaction_factor) - n
    )


# -----------------------------------------------------------------------------
# ProxySPEX adapter integration (_select_odd_interactions)
# -----------------------------------------------------------------------------


def _fit_surrogate(approx, game, budget):
    approx._sampler.sample(budget)
    coalitions = approx._sampler.coalitions_matrix
    game_values = np.asarray(game(coalitions), dtype=float)
    surrogate = approx._fit_surrogate_model(
        coalitions=coalitions,
        game_values=game_values,
    )
    return coalitions, game_values, surrogate


def test_select_odd_interactions_returns_higher_order_odd_only():
    n = 10
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    *_, surrogate = _fit_surrogate(approx, game, 2**n)
    selected = approx._select_odd_interactions(
        n_candidate_interactions=10,
        surrogate_model=surrogate,
    )
    # Guard against a silent break of convert_tree_model / _sklearn_to_fourier
    # that would make screening return nothing and pass the loop vacuously.
    assert len(selected) > 0
    for t in selected:
        assert isinstance(t, tuple)
        assert len(t) >= 3
        assert len(t) % 2 == 1


def test_select_odd_interactions_respects_k():
    n = 10
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    *_, surrogate = _fit_surrogate(approx, game, 2**n)
    k = 5
    selected = approx._select_odd_interactions(
        n_candidate_interactions=k,
        surrogate_model=surrogate,
    )
    assert len(selected) <= k


def test_select_odd_interactions_handles_zero_budget():
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    *_, surrogate = _fit_surrogate(approx, game, 2**n)
    assert (
        approx._select_odd_interactions(
            n_candidate_interactions=0,
            surrogate_model=surrogate,
        )
        == []
    )


def test_select_odd_interactions_handles_missing_surrogate():
    n = 8
    approx = OddSHAP(n=n, random_state=0)
    assert (
        approx._select_odd_interactions(
            n_candidate_interactions=10,
            surrogate_model=None,
        )
        == []
    )


def test_full_budget_bypasses_candidate_truncation(monkeypatch):
    """At budget = 2**n the candidate support must not be truncated to
    ceil(budget / interaction_factor): ``approximate`` passes the untruncated
    candidate count (2**n) down to the screening step."""
    n = 8
    approx = OddSHAP(n=n, random_state=0)
    captured = {}

    def additive_game(coalitions):
        return coalitions.astype(float).sum(axis=1)

    def fake_select_odd_interactions(**kwargs):
        captured["n_candidate_interactions"] = kwargs["n_candidate_interactions"]
        return []

    monkeypatch.setattr(approx, "_fit_surrogate_model", lambda **kw: object())
    monkeypatch.setattr(approx, "_select_odd_interactions", fake_select_odd_interactions)

    approx.approximate(2**n, additive_game)

    assert captured["n_candidate_interactions"] == 2**n
    assert captured["n_candidate_interactions"] > math.ceil(2**n / approx.interaction_factor)


def test_select_odd_interactions_returns_full_support_for_large_k():
    """With a candidate budget larger than the available odd support, the
    screening returns the entire higher-order odd support (no spurious cap)."""
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    *_, surrogate = _fit_surrogate(approx, game, 2**n)
    selected_small = approx._select_odd_interactions(
        n_candidate_interactions=3,
        surrogate_model=surrogate,
    )
    selected_all = approx._select_odd_interactions(
        n_candidate_interactions=2**n,
        surrogate_model=surrogate,
    )
    assert len(selected_small) <= 3
    assert len(selected_all) >= len(selected_small)
    assert all(len(t) >= 3 and len(t) % 2 == 1 for t in selected_all)


# -----------------------------------------------------------------------------
# `_build_support` invariants
# -----------------------------------------------------------------------------


def test_build_support_always_includes_empty_and_singletons():
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support([(0, 1, 2)])
    assert () in approx.odd_interaction_lookup
    for i in range(n):
        assert (i,) in approx.odd_interaction_lookup
    assert (0, 1, 2) in approx.odd_interaction_lookup


def test_build_support_drops_singleton_inputs():
    """_build_support skips singleton inputs because singletons are always included."""
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support([(0,), (0, 1, 2, 3, 4)])
    higher_order = {t for t in approx.odd_interaction_lookup if len(t) >= 2}
    assert higher_order == {(0, 1, 2, 3, 4)}


def test_build_support_normalizes_unsorted_tuples():
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support([(2, 0, 1)])  # unsorted input
    assert (0, 1, 2) in approx.odd_interaction_lookup
    assert (2, 0, 1) not in approx.odd_interaction_lookup


def test_build_support_n_active_interactions_consistent():
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support([(0, 1, 2), (1, 3, 5)])
    expected = 1 + n + 2  # empty + n singletons + 2 higher-order odd
    assert approx.n_active_interactions == expected
    assert approx.odd_interaction_matrix_binary.shape == (expected, n)


# -----------------------------------------------------------------------------
# Game-property tests (Shapley axioms on closed-form games)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 8])
def test_dummy_game_singleton_attribution(n):
    """DummyGame((1, 2)) attributes value ~0.5 each to players 1 and 2 by symmetry."""
    interaction = (1, 2)
    game = DummyGame(n=n, interaction=interaction)
    approx = OddSHAP(n=n, random_state=0)
    iv = approx.approximate(2**n, game)
    v_full = float(game(np.ones((1, n), dtype=bool))[0])
    v_empty = float(game(np.zeros((1, n), dtype=bool))[0])
    # efficiency holds exactly:
    assert np.sum(iv.values[1:]) == pytest.approx(v_full - v_empty, abs=1e-6)
    # by symmetry, players 1 and 2 should have identical Shapley values
    assert iv.values[2] == pytest.approx(iv.values[3], abs=1e-3)


# -----------------------------------------------------------------------------
# Convergence vs ExactComputer
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 8])
def test_convergence_vs_exact_at_full_budget(n):
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    iv = approx.approximate(2**n, game)
    exact = ExactComputer(game, n_players=n)(index="SV")
    np.testing.assert_allclose(iv.values, exact.values, atol=1e-2)


@pytest.mark.parametrize("n", [6, 8])
def test_efficiency_persists_at_sub_budget(n):
    """Efficiency axiom is enforced by construction, so it must hold even sub-budget."""
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    budget = max(n * approx.interaction_factor, int(0.5 * 2**n))
    iv = approx.approximate(budget, game)
    v_full = float(game(np.ones((1, n), dtype=bool))[0])
    v_empty = float(game(np.zeros((1, n), dtype=bool))[0])
    assert np.sum(iv.values[1:]) == pytest.approx(v_full - v_empty, abs=1e-6)


# -----------------------------------------------------------------------------
# Remaining branch coverage
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1])
def test_sampling_weights_reject_degenerate_n(n):
    with pytest.raises(ValueError, match="undefined for n <= 1"):
        OddSHAP._init_sampling_weights_static(n)


@pytest.mark.parametrize("n", [0, 1])
def test_regression_kernel_weights_reject_degenerate_n(n):
    with pytest.raises(ValueError, match="undefined for n <= 1"):
        OddSHAP._init_regression_kernel_weights_static(n)


@skip_if_no_lightgbm
def test_custom_tree_params_are_used_for_the_surrogate_fit():
    """The tree_params branch of `_fit_surrogate_model` builds the LightGBM
    surrogate from the user-supplied parameters."""
    n = 6
    game = DummyGame(n=n, interaction=(1, 2))
    approx = OddSHAP(n=n, random_state=0, tree_params={"max_depth": 2, "n_estimators": 5})
    approx._sampler.sample(2**n)
    coalitions = approx._sampler.coalitions_matrix
    game_values = np.asarray(game(coalitions), dtype=float)
    surrogate = approx._fit_surrogate_model(coalitions=coalitions, game_values=game_values)
    assert surrogate.get_params()["max_depth"] == 2
    assert surrogate.get_params()["n_estimators"] == 5


@skip_if_no_lightgbm
def test_tree_params_can_override_shared_surrogate_kwargs():
    """tree_params entries override the built-in surrogate kwargs without a TypeError."""
    n = 6
    game = DummyGame(n=n, interaction=(1, 2))
    approx = OddSHAP(n=n, random_state=0, tree_params={"random_state": 7, "n_jobs": 2})
    approx._sampler.sample(2**n)
    coalitions = approx._sampler.coalitions_matrix
    game_values = np.asarray(game(coalitions), dtype=float)
    surrogate = approx._fit_surrogate_model(coalitions=coalitions, game_values=game_values)
    assert surrogate.get_params()["random_state"] == 7
    assert surrogate.get_params()["n_jobs"] == 2
    assert surrogate.get_params()["max_depth"] == 10


@skip_if_no_lightgbm
def test_tree_params_without_depth_keeps_paper_default():
    """tree_params lacking max_depth keeps the paper's depth-10 surrogate (not unlimited)."""
    n = 6
    game = DummyGame(n=n, interaction=(1, 2))
    approx = OddSHAP(n=n, random_state=0, tree_params={"n_estimators": 5})
    approx._sampler.sample(2**n)
    coalitions = approx._sampler.coalitions_matrix
    game_values = np.asarray(game(coalitions), dtype=float)
    surrogate = approx._fit_surrogate_model(coalitions=coalitions, game_values=game_values)
    assert surrogate.get_params()["max_depth"] == 10
    assert surrogate.get_params()["n_estimators"] == 5


def test_build_support_with_none_yields_baseline_support():
    """`_build_support(None)` produces the minimal support: () plus singletons."""
    n = 5
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support(None)
    assert approx.n_active_interactions == n + 1
    assert list(approx.odd_interaction_lookup) == [()] + [(i,) for i in range(n)]


def test_build_weighted_system_keeps_boundary_rows_when_requested():
    """`drop_boundary_rows=False` keeps the empty and grand coalition rows."""
    n = 5
    game = DummyGame(n=n, interaction=(1, 2))
    approx = OddSHAP(n=n, random_state=0)
    approx._sampler.sample(2**n)
    coalitions = approx._sampler.coalitions_matrix
    game_values = np.asarray(game(coalitions), dtype=float)
    approx._build_support(None)
    x_drop, _y_drop = approx._build_weighted_system(
        coalitions=coalitions,
        game_values=game_values,
        empty_set_value=0.0,
        full_set_value=1.0,
        drop_boundary_rows=True,
    )
    x_keep, y_keep = approx._build_weighted_system(
        coalitions=coalitions,
        game_values=game_values,
        empty_set_value=0.0,
        full_set_value=1.0,
        drop_boundary_rows=False,
    )
    assert x_keep.shape[0] == coalitions.shape[0]
    assert x_keep.shape[0] == x_drop.shape[0] + 2
    assert y_keep.shape[0] == x_keep.shape[0]


# -----------------------------------------------------------------------------
# Defensive guards (contract violations / invalid internal calls)
# -----------------------------------------------------------------------------


def test_missing_empty_coalition_raises():
    """If the sampler omits the empty coalition (contract violation), approximate raises."""
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    fake_sampler = MagicMock()
    fake_sampler.coalitions_matrix = np.ones((1, n), dtype=bool)
    fake_sampler.empty_coalition_index = None
    approx._sampler = fake_sampler
    with pytest.raises(RuntimeError, match="expected empty coalition"):
        approx.approximate(n * approx.interaction_factor, lambda c: np.zeros(len(c)))


def test_missing_grand_coalition_raises():
    """If the sampler omits the grand coalition (contract violation), approximate raises."""
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    fake_sampler = MagicMock()
    fake_sampler.coalitions_matrix = np.zeros((1, n), dtype=bool)
    fake_sampler.empty_coalition_index = 0
    approx._sampler = fake_sampler
    with pytest.raises(RuntimeError, match="expected grand coalition"):
        approx.approximate(n * approx.interaction_factor, lambda c: np.zeros(len(c)))


def test_build_weighted_system_without_support_raises():
    """_build_weighted_system requires _build_support to have run first."""
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    with pytest.raises(RuntimeError, match="support has not been built"):
        approx._build_weighted_system(
            coalitions=np.zeros((2, n), dtype=bool),
            game_values=np.zeros(2),
            empty_set_value=0.0,
            full_set_value=1.0,
        )


def test_build_constraint_system_without_support_raises():
    """_build_constraint_system requires a non-empty support."""
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    with pytest.raises(RuntimeError, match="at least one non-empty interaction"):
        approx._build_constraint_system(full_set_value=1.0, empty_set_value=0.0)


def test_transform_to_shapley_rejects_wrong_length():
    """_transform_to_shapley validates the coefficient-vector length against the support."""
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support([(0, 1, 2)])
    wrong = np.zeros(approx.n_active_interactions + 1)
    with pytest.raises(ValueError, match="does not match the active OddSHAP support"):
        approx._transform_to_shapley(wrong, baseline_value=0.0)


def test_transform_to_shapley_vectorized_matches_formula():
    """The vectorized Shapley transform matches the expected closed-form values."""
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support([(0, 1, 2)])
    coeffs = np.ones(approx.n_active_interactions)
    sv = approx._transform_to_shapley(coeffs, baseline_value=0.0)
    # singletons contribute -2 * 1/1 = -2; the triple contributes -2 * 1/3
    for i in range(n):
        expected = -2.0 * 1.0  # singleton (i,)
        if i in (0, 1, 2):
            expected += -2.0 * (1.0 / 3.0)  # triple (0,1,2)
        np.testing.assert_allclose(sv[i + 1], expected, atol=1e-12)
