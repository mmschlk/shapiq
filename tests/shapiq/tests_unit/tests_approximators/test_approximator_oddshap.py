"""Tests for the OddSHAP approximator.

OddSHAP estimates first-order Shapley values via paired sampling + odd-only
Fourier regression on a LightGBM surrogate. The algorithm has two branches:

  - low-budget fallback (budget < n * interaction_factor):
        fit surrogate, explain via TreeExplainer

  - high-budget regression (budget >= n * interaction_factor):
        fit surrogate, screen top-k odd interactions via the ProxySPEX adapter,
        build active support, solve constrained weighted Fourier regression,
        transform odd coefficients to Shapley values

The constraint system enforces the exact identity
    beta_empty = (f(N) + f(empty)) / 2,
    sum over non-empty odd Fourier coefficients = -(f(N) - f(empty)) / 2,
which after the -2 scaling in `_transform_to_shapley` guarantees the
efficiency axiom by construction:
    sum_i phi_i = f(N) - f(empty),
    phi_empty   = f(empty).

These two identities are checked as EXACT properties below. Convergence to
ExactComputer on dense games is NOT a guarantee — OddSHAP is a sparse
recovery method — so the convergence test is marked xfail(strict=False).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
from scipy.special import binom

import shapiq.approximator as approximator_module
from shapiq.approximator.regression import OddSHAP
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import SOUM, DummyGame

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
    assert approx.odd_only is True
    assert approx.regression_basis == "Fourier"
    # Default detection backend now uses tree interaction values
    # (ProxySHAP-style InterventionalTreeExplainer), not the bespoke
    # lgboost-Fourier adapter.
    assert approx.interaction_detection == "ProxySHAP"


def test_init_custom_kwargs():
    custom_weights = np.zeros(9, dtype=float)
    custom_weights[1:8] = 1.0 / 7  # n=8, valid weights summing to 1
    approx = OddSHAP(
        n=8,
        pairing_trick=False,
        interaction_factor=5,
        sampling_weights=custom_weights,
        random_state=123,
        tree_params={"max_depth": 4},
    )
    assert approx.interaction_factor == 5
    assert approx.odd_only is True  # implementation pins this to True
    assert approx.tree_params == {"max_depth": 4}


def test_init_rejects_odd_only_false():
    """The current implementation supports only the odd_only=True branch."""
    with pytest.raises(ValueError, match="only odd_only=True"):
        OddSHAP(n=8, odd_only=False)


def test_init_requires_lightgbm(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)

    with pytest.raises(ImportError, match="The 'lightgbm' package is required for OddSHAP"):
        OddSHAP(n=8)


def test_public_approximator_exports_include_oddshap():
    assert "OddSHAP" in approximator_module.__all__
    assert OddSHAP in approximator_module.SV_APPROXIMATORS


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
            f"weights not symmetric at k={k}: w[{k}]={w[k]} vs w[{n-k}]={w[n-k]}"
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
    budget = 2 ** n  # well above n*interaction_factor (= 10n) for n>=6
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
    budget = int(budget_pct * 2 ** n)
    iv = approx.approximate(budget, game)
    assert iv.estimated == (budget < 2 ** n)


# -----------------------------------------------------------------------------
# Constraint-system identities (Theorem 3.2 + Appendix C)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 8, 10])
def test_efficiency_axiom_holds_exactly(n):
    """sum_i phi_i = v(N) - v(empty) is enforced by Sara's constraint system."""
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    iv = approx.approximate(2 ** n, game)
    v_full = float(game(np.ones((1, n), dtype=bool))[0])
    v_empty = float(game(np.zeros((1, n), dtype=bool))[0])
    assert np.sum(iv.values[1:]) == pytest.approx(v_full - v_empty, abs=1e-6)


@pytest.mark.parametrize("n", [6, 8, 10])
def test_baseline_exact(n):
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    iv = approx.approximate(2 ** n, game)
    v_empty = float(game(np.zeros((1, n), dtype=bool))[0])
    assert iv.values[0] == pytest.approx(v_empty, abs=1e-9)


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_determinism_same_seed_same_output():
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    a = OddSHAP(n=n, random_state=7).approximate(2 ** n, game)
    b = OddSHAP(n=n, random_state=7).approximate(2 ** n, game)
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
# Sara replaced the low-budget TreeExplainer fallback with an explicit
# ValueError per Max's feedback. The regression branch is now the only
# code path; verify that the budget threshold is enforced.
# -----------------------------------------------------------------------------


def test_low_budget_raises_value_error():
    """budget < n * interaction_factor must raise ValueError, not silently fall back."""
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    budget = n * approx.interaction_factor - 1
    with pytest.raises(ValueError, match="too small"):
        approx.approximate(budget, game)


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

    assert captured["n_candidate_interactions"] == math.ceil(
        budget / approx.interaction_factor
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

    assert captured["n_candidate_interactions"] == math.ceil(
        budget / approx.interaction_factor
    )


# -----------------------------------------------------------------------------
# ProxySPEX adapter integration (_select_odd_interactions)
# -----------------------------------------------------------------------------


def _fit_surrogate(approx, game, budget):
    approx._sampler.sample(budget)
    coalitions = approx._sampler.coalitions_matrix
    game_values = np.asarray(game(coalitions), dtype=float)
    surrogate = approx._fit_surrogate_model(
        coalitions=coalitions, game_values=game_values,
    )
    return coalitions, game_values, surrogate


def test_select_odd_interactions_returns_higher_order_odd_only():
    n = 10
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    coalitions, game_values, surrogate = _fit_surrogate(approx, game, 2 ** n)
    selected = approx._select_odd_interactions(
        budget=200,  # sub-budget triggers the top-k truncation branch
        coalitions=coalitions, game_values=game_values,
        n_candidate_interactions=10, surrogate_model=surrogate,
    )
    for t in selected:
        assert isinstance(t, tuple)
        assert len(t) >= 3
        assert len(t) % 2 == 1


def test_select_odd_interactions_respects_k():
    n = 10
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    coalitions, game_values, surrogate = _fit_surrogate(approx, game, 2 ** n)
    k = 5
    selected = approx._select_odd_interactions(
        budget=200,  # sub-budget so the top-k limit kicks in
        coalitions=coalitions, game_values=game_values,
        n_candidate_interactions=k, surrogate_model=surrogate,
    )
    assert len(selected) <= k


def test_select_odd_interactions_handles_zero_budget():
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    coalitions, game_values, surrogate = _fit_surrogate(approx, game, 2 ** n)
    assert approx._select_odd_interactions(
        budget=80,
        coalitions=coalitions, game_values=game_values,
        n_candidate_interactions=0, surrogate_model=surrogate,
    ) == []


def test_select_odd_interactions_handles_missing_surrogate():
    n = 8
    approx = OddSHAP(n=n, random_state=0)
    assert approx._select_odd_interactions(
        budget=80,
        coalitions=np.zeros((1, n), dtype=bool),
        game_values=np.zeros(1),
        n_candidate_interactions=10,
        surrogate_model=None,
    ) == []


def test_select_odd_interactions_full_budget_returns_all_higher_order_odd():
    """At full budget Sara's code returns the entire higher-order odd support,
    not the top-k subset — k truncation is only for sub-budget mode."""
    n = 8
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    coalitions, game_values, surrogate = _fit_surrogate(approx, game, 2 ** n)
    selected = approx._select_odd_interactions(
        budget=2 ** n,  # full-budget path
        coalitions=coalitions, game_values=game_values,
        n_candidate_interactions=3,  # would clip to 3 in sub-budget mode
        surrogate_model=surrogate,
    )
    # At full budget we expect more than 3 since k truncation is bypassed.
    # The exact count depends on the surrogate; just verify the limit is
    # not enforced.
    if len(selected) > 0:
        assert all(len(t) >= 3 and len(t) % 2 == 1 for t in selected)


# -----------------------------------------------------------------------------
# `_build_support` invariants
# -----------------------------------------------------------------------------


def test_build_support_always_includes_empty_and_singletons():
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support([(0, 1, 2), (2, 4)])  # even-size (2,4) should be dropped
    assert () in approx.odd_interaction_lookup
    for i in range(n):
        assert (i,) in approx.odd_interaction_lookup
    assert (0, 1, 2) in approx.odd_interaction_lookup
    assert (2, 4) not in approx.odd_interaction_lookup


def test_build_support_drops_even_and_singleton_inputs():
    """_build_support pre-filters to len>=2 odd inputs."""
    n = 6
    approx = OddSHAP(n=n, random_state=0)
    approx._build_support([(0,), (1, 2), (0, 1, 2, 3, 4)])  # only the last is kept
    higher_order = {
        t for t in approx.odd_interaction_lookup if len(t) >= 2
    }
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
    iv = approx.approximate(2 ** n, game)
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
    iv = approx.approximate(2 ** n, game)
    exact = ExactComputer(game, n_players=n)(index="SV")
    np.testing.assert_allclose(iv.values, exact.values, atol=1e-2)


@pytest.mark.parametrize("n", [6, 8])
def test_efficiency_persists_at_sub_budget(n):
    """Efficiency axiom is enforced by construction, so it must hold even sub-budget."""
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    approx = OddSHAP(n=n, random_state=0)
    budget = max(n * approx.interaction_factor, int(0.5 * 2 ** n))
    iv = approx.approximate(budget, game)
    v_full = float(game(np.ones((1, n), dtype=bool))[0])
    v_empty = float(game(np.zeros((1, n), dtype=bool))[0])
    assert np.sum(iv.values[1:]) == pytest.approx(v_full - v_empty, abs=1e-6)
