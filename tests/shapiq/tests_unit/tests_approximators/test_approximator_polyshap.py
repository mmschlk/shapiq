"""Tests for the PolySHAP approximators: PolySHAPKAdd, PolySHAPPartial, PolySHAPPrior."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import binom

from shapiq.approximator.regression import PolySHAPKAdd, PolySHAPPartial, PolySHAPPrior
from shapiq.approximator.regression.polyshap.polyshap import PolySHAP
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset
from shapiq_games.synthetic import DummyGame

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _singleton_prior(n: int) -> list[tuple]:
    """Minimal prior: empty set + all singletons (structurally identical to KernelSHAP frontier)."""
    return list(powerset(range(n), max_size=1))


def _kadd_frontier_size(n: int, max_order: int) -> int:
    return int(sum(binom(n, k) for k in range(max_order + 1)))


# ---------------------------------------------------------------------------
# PolySHAP base-class: frontier validation
# ---------------------------------------------------------------------------

def test_polyshap_raises_when_singleton_missing():
    """PolySHAP must reject a frontier that omits any singleton."""
    n = 4
    incomplete = {(): 0, (0,): 1, (1,): 2}  # players 2 and 3 missing
    with pytest.raises(ValueError, match="main effects"):
        PolySHAP(n, incomplete)


def test_polyshap_accepts_valid_custom_frontier():
    """PolySHAP should accept any frontier that contains all singletons."""
    n = 3
    frontier = {(): 0, (0,): 1, (1,): 2, (2,): 3, (0, 1): 4}
    approx = PolySHAP(n, frontier)
    assert approx.n == n
    assert len(approx.explanation_frontier) == len(frontier)
    assert approx.n_variables == len(frontier) - 1


def test_polyshap_interaction_matrix_shape():
    """interaction_matrix_binary must have shape (|frontier|, n)."""
    n = 4
    frontier = {S: pos for pos, S in enumerate(powerset(range(n), max_size=2))}
    approx = PolySHAP(n, frontier)
    assert approx.interaction_matrix_binary.shape == (len(frontier), n)


# ---------------------------------------------------------------------------
# PolySHAPKAdd – initialisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "max_order"),
    [(3, 1), (3, 2), (7, 1), (7, 2), (10, 3)],
)
def test_kadd_init_attributes(n, max_order):
    """PolySHAPKAdd must expose correct attributes after construction."""
    approx = PolySHAPKAdd(n, max_order)
    expected_size = _kadd_frontier_size(n, max_order)
    assert approx.n == n
    assert approx.max_order == 1       # PolySHAP always targets SVs
    assert approx.min_order == 0
    assert approx.top_order is False
    assert approx.iteration_cost == 1
    assert approx.index == "SV"
    assert len(approx.explanation_frontier) == expected_size
    assert approx.n_variables == expected_size - 1


@pytest.mark.parametrize("n", [4, 7])
def test_kadd_frontier_contains_all_singletons(n):
    """Every singleton (i,) must be a key in the explanation frontier."""
    approx = PolySHAPKAdd(n, max_order=2)
    for i in range(n):
        assert (i,) in approx.explanation_frontier


@pytest.mark.parametrize("excluded_size", [2, 3])
def test_kadd_sizes_to_exclude_removes_coalitions(excluded_size):
    """Coalitions of an excluded size must be absent from the frontier."""
    approx = PolySHAPKAdd(6, max_order=4, sizes_to_exclude={excluded_size})
    for coalition in approx.explanation_frontier:
        assert len(coalition) != excluded_size


def test_kadd_sizes_to_exclude_singletons_raises():
    """Excluding size 1 (singletons) must propagate a ValueError from the base class."""
    with pytest.raises(ValueError, match="main effects"):
        PolySHAPKAdd(4, max_order=2, sizes_to_exclude={1})


def test_kadd_frontier_positions_are_unique():
    """Each frontier term must map to a unique column position."""
    approx = PolySHAPKAdd(5, max_order=2)
    positions = list(approx.explanation_frontier.values())
    assert len(positions) == len(set(positions))


# ---------------------------------------------------------------------------
# PolySHAPKAdd – approximation quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("n", "budget"), [(7, 500), (7, 100)])
def test_kadd_approximate_sv_quality(n, budget):
    """PolySHAPKAdd(max_order=1) recovers correct SVs on DummyGame."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approx = PolySHAPKAdd(n, max_order=1, random_state=42)
    sv = approx.approximate(budget, game)

    assert isinstance(sv, InteractionValues)
    assert sv.index == "SV"
    assert sv.max_order == 1
    assert sv.min_order == 0
    assert sv.estimation_budget <= budget
    assert sv.estimated != (budget >= 2**n)
    assert game.access_counter <= budget

    # Players 1 and 2 carry the interaction: SV ≈ 0.6429
    assert sv[(1,)] == pytest.approx(0.6429, abs=0.1)
    assert sv[(2,)] == pytest.approx(0.6429, abs=0.1)

    # Efficiency: sum of SVs equals v(N) - v({}) = 2.0
    assert np.sum(sv.values) == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_kadd_approximate_higher_order_converges(max_order):
    """PolySHAPKAdd at any order still converges to the correct SVs on DummyGame."""
    n, budget = 7, 400
    game = DummyGame(n, (1, 2))
    sv = PolySHAPKAdd(n, max_order=max_order, random_state=0).approximate(budget, game)
    assert sv[(1,)] == pytest.approx(0.6429, abs=0.15)
    assert sv[(2,)] == pytest.approx(0.6429, abs=0.15)
    assert np.sum(sv.values) == pytest.approx(2.0, abs=0.15)


def test_kadd_all_players_present_in_output():
    """The output must contain a Shapley value for every player."""
    n = 7
    game = DummyGame(n, (1, 2))
    sv = PolySHAPKAdd(n, max_order=1, random_state=0).approximate(200, game)
    for i in range(n):
        assert (i,) in sv.interaction_lookup


def test_kadd_pairing_trick_produces_valid_output():
    """PolySHAPKAdd with pairing_trick=True must still produce valid SV estimates."""
    n, budget = 7, 200
    game = DummyGame(n, (1, 2))
    sv = PolySHAPKAdd(n, max_order=1, pairing_trick=True, random_state=7).approximate(budget, game)
    assert isinstance(sv, InteractionValues)
    assert sv[(1,)] == pytest.approx(0.6429, abs=0.15)


def test_kadd_random_state_reproducibility():
    """Identical random_state must yield bit-identical approximations."""
    n, budget = 7, 200
    sv1 = PolySHAPKAdd(n, max_order=1, random_state=99).approximate(budget, DummyGame(n, (1, 2)))
    sv2 = PolySHAPKAdd(n, max_order=1, random_state=99).approximate(budget, DummyGame(n, (1, 2)))
    np.testing.assert_array_equal(sv1.values, sv2.values)


def test_kadd_not_estimated_when_full_space_covered():
    """estimated must be False when budget exceeds the full coalition space."""
    n = 5
    sv = PolySHAPKAdd(n, max_order=1, random_state=0).approximate(2**n + 10, DummyGame(n, (1, 2)))
    assert sv.estimated is False


def test_kadd_estimated_when_budget_small():
    """estimated must be True when budget is smaller than the full coalition space."""
    n = 7
    sv = PolySHAPKAdd(n, max_order=1, random_state=0).approximate(50, DummyGame(n, (1, 2)))
    assert sv.estimated is True


# ---------------------------------------------------------------------------
# PolySHAPPartial – initialisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "n_terms"),
    [
        (7, 8),   # exactly empty + singletons, no higher-order extension
        (7, 12),  # singletons + 4 pairs
        (7, 25),  # singletons + some triples (when pairs exhausted earlier)
    ],
)
def test_partial_frontier_size(n, n_terms):
    """PolySHAPPartial must build a frontier with exactly n_explanation_terms entries."""
    approx = PolySHAPPartial(n, n_explanation_terms=n_terms, random_state=42)
    assert len(approx.explanation_frontier) == n_terms
    assert approx.n_variables == n_terms - 1


def test_partial_always_contains_all_singletons():
    """All singletons must appear in the frontier regardless of n_explanation_terms."""
    n = 7
    approx = PolySHAPPartial(n, n_explanation_terms=15, random_state=1)
    for i in range(n):
        assert (i,) in approx.explanation_frontier


def test_partial_different_seeds_produce_different_frontiers():
    """Different random states must yield different higher-order interaction sets."""
    n, n_terms = 7, 20
    a1 = PolySHAPPartial(n, n_terms, random_state=1)
    a2 = PolySHAPPartial(n, n_terms, random_state=2)
    assert set(a1.explanation_frontier) != set(a2.explanation_frontier)


def test_partial_same_seed_reproduces_frontier():
    """Same random state must yield an identical frontier."""
    n, n_terms = 7, 20
    a1 = PolySHAPPartial(n, n_terms, random_state=7)
    a2 = PolySHAPPartial(n, n_terms, random_state=7)
    assert set(a1.explanation_frontier) == set(a2.explanation_frontier)


def test_partial_sizes_to_exclude_omits_those_sizes():
    """Coalitions of excluded sizes must not appear in the extended frontier."""
    n, n_terms = 7, 25
    approx = PolySHAPPartial(n, n_terms, sizes_to_exclude={2}, random_state=0)
    for coalition in approx.explanation_frontier:
        assert len(coalition) != 2


# ---------------------------------------------------------------------------
# PolySHAPPartial – approximation quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("n", "budget"), [(7, 400), (7, 120)])
def test_partial_approximate_sv_quality(n, budget):
    """PolySHAPPartial with a minimal frontier approximates SVs correctly on DummyGame."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approx = PolySHAPPartial(n, n_explanation_terms=n + 1, random_state=42)
    sv = approx.approximate(budget, game)

    assert isinstance(sv, InteractionValues)
    assert sv.index == "SV"
    assert sv.estimation_budget <= budget
    assert game.access_counter <= budget

    assert sv[(1,)] == pytest.approx(0.6429, abs=0.1)
    assert sv[(2,)] == pytest.approx(0.6429, abs=0.1)
    assert np.sum(sv.values) == pytest.approx(2.0, abs=0.1)


def test_partial_extended_frontier_approximates_correctly():
    """PolySHAPPartial with higher-order terms still recovers correct SVs."""
    n, budget = 7, 400
    game = DummyGame(n, (1, 2))
    approx = PolySHAPPartial(n, n_explanation_terms=20, random_state=42)
    sv = approx.approximate(budget, game)
    assert sv[(1,)] == pytest.approx(0.6429, abs=0.15)
    assert sv[(2,)] == pytest.approx(0.6429, abs=0.15)


# ---------------------------------------------------------------------------
# PolySHAPPrior – initialisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 7, 10])
def test_prior_init_attributes(n):
    """PolySHAPPrior must expose correct attributes for a singleton prior."""
    prior = _singleton_prior(n)
    approx = PolySHAPPrior(n, q_prior=prior)
    assert approx.n == n
    assert approx.max_order == 1
    assert approx.min_order == 0
    assert approx.top_order is False
    assert approx.index == "SV"
    assert len(approx.explanation_frontier) == len(prior)
    assert approx.n_variables == len(prior) - 1


def test_prior_raises_when_singleton_missing():
    """PolySHAPPrior must propagate a ValueError for a prior that omits singletons."""
    n = 5
    incomplete = [(), (0,), (1,), (2,)]  # players 3 and 4 absent
    with pytest.raises(ValueError, match="main effects"):
        PolySHAPPrior(n, q_prior=incomplete)


@pytest.mark.parametrize("n", [3, 7])
def test_prior_frontier_positions_match_enumeration_order(n):
    """Each coalition's position in the frontier must match its index in q_prior."""
    prior = _singleton_prior(n) + [(0, 1), (1, 2)]
    approx = PolySHAPPrior(n, q_prior=prior)
    for pos, coalition in enumerate(prior):
        assert approx.explanation_frontier[coalition] == pos


# ---------------------------------------------------------------------------
# PolySHAPPrior – approximation quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("n", "budget"), [(7, 500), (7, 100)])
def test_prior_approximate_sv_quality(n, budget):
    """PolySHAPPrior with a singleton prior estimates SVs correctly on DummyGame."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    approx = PolySHAPPrior(n, q_prior=_singleton_prior(n), random_state=42)
    sv = approx.approximate(budget, game)

    assert isinstance(sv, InteractionValues)
    assert sv.index == "SV"
    assert sv.estimation_budget <= budget
    assert sv.estimated != (budget >= 2**n)
    assert game.access_counter <= budget

    assert sv[(1,)] == pytest.approx(0.6429, abs=0.1)
    assert sv[(2,)] == pytest.approx(0.6429, abs=0.1)
    assert np.sum(sv.values) == pytest.approx(2.0, abs=0.1)


def test_prior_with_interaction_pairs_still_correct():
    """Adding interaction pairs to the prior must not break SV accuracy."""
    n, budget = 7, 400
    game = DummyGame(n, (1, 2))
    prior = _singleton_prior(n) + [(1, 2), (0, 3), (2, 4)]
    approx = PolySHAPPrior(n, q_prior=prior, random_state=42)
    sv = approx.approximate(budget, game)
    assert sv[(1,)] == pytest.approx(0.6429, abs=0.15)
    assert sv[(2,)] == pytest.approx(0.6429, abs=0.15)


def test_prior_agrees_exactly_with_kadd_order1():
    """PolySHAPPrior (singleton prior) and PolySHAPKAdd(max_order=1) share the same frontier
    and implementation path, so they must produce bit-identical SVs given the same random state."""
    n, budget, seed = 7, 200, 5
    sv_prior = PolySHAPPrior(
        n, q_prior=_singleton_prior(n), random_state=seed
    ).approximate(budget, DummyGame(n, (1, 2)))
    sv_kadd = PolySHAPKAdd(
        n, max_order=1, random_state=seed
    ).approximate(budget, DummyGame(n, (1, 2)))
    np.testing.assert_array_almost_equal(sv_prior.values, sv_kadd.values, decimal=12)


# ---------------------------------------------------------------------------
# Shared behaviour across approximators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("ApproxClass", "kwargs"),
    [
        (PolySHAPKAdd,  {"max_order": 1}),
        (PolySHAPKAdd,  {"max_order": 2}),
        (PolySHAPPartial, {"n_explanation_terms": 10}),
        (PolySHAPPrior, {"q_prior": _singleton_prior(7)}),
    ],
)
def test_budget_never_exceeded(ApproxClass, kwargs):
    """The game access counter must never exceed the declared budget."""
    n, budget = 7, 150
    game = DummyGame(n, (1, 2))
    ApproxClass(n, **kwargs, random_state=0).approximate(budget, game)
    assert game.access_counter <= budget


@pytest.mark.parametrize(
    ("ApproxClass", "kwargs"),
    [
        (PolySHAPKAdd,  {"max_order": 1}),
        (PolySHAPKAdd,  {"max_order": 2}),
        (PolySHAPPartial, {"n_explanation_terms": 10}),
        (PolySHAPPrior, {"q_prior": _singleton_prior(7)}),
    ],
)
def test_sv_efficiency_with_large_budget(ApproxClass, kwargs):
    """Sum of estimated Shapley values must equal v(N) - v({}) ≈ 2.0 for DummyGame(7, (1,2))."""
    n = 7
    game = DummyGame(n, (1, 2))
    sv = ApproxClass(n, **kwargs, random_state=0).approximate(500, game)
    assert np.sum(sv.values) == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize(
    ("ApproxClass", "kwargs"),
    [
        (PolySHAPKAdd,  {"max_order": 1}),
        (PolySHAPPartial, {"n_explanation_terms": 8}),
        (PolySHAPPrior, {"q_prior": _singleton_prior(7)}),
    ],
)
def test_index_is_sv(ApproxClass, kwargs):
    """All PolySHAP approximators must report index='SV'."""
    n = 7
    sv = ApproxClass(n, **kwargs, random_state=0).approximate(200, DummyGame(n, (1, 2)))
    assert sv.index == "SV"


@pytest.mark.parametrize(
    ("ApproxClass", "kwargs"),
    [
        (PolySHAPKAdd,  {"max_order": 1}),
        (PolySHAPPartial, {"n_explanation_terms": 8}),
        (PolySHAPPrior, {"q_prior": _singleton_prior(7)}),
    ],
)
def test_all_players_have_sv_in_output(ApproxClass, kwargs):
    """The output InteractionValues must contain an SV entry for each player."""
    n = 7
    sv = ApproxClass(n, **kwargs, random_state=0).approximate(200, DummyGame(n, (1, 2)))
    for i in range(n):
        assert (i,) in sv.interaction_lookup
