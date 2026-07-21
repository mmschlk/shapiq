"""Tests for the :class:`~shapiq.approximator.PolySHAP` approximator.

PolySHAP selects its interaction frontier through one of three constructor arguments:

* **k-additive** (default): ``max_order`` (with ``max_terms=None``).
* **partial**: ``max_terms`` (budget-capped frontier, whole low orders first).
* **prior**: ``prior_frontier`` supplies the interaction terms directly.

The suite exercises every branch of the frontier builder and both design-matrix
paths of :meth:`~shapiq.approximator.PolySHAP.approximate` (the order-1 KernelSHAP path
and the higher-order interaction path), checks convergence to the ground truth from
:class:`~shapiq.ExactComputer`, and runs reproducibility / variance sweeps over a
fixed list of 50 seeds.
"""

from __future__ import annotations

from functools import cache
from itertools import pairwise

import numpy as np
import pytest
from scipy.special import binom

import shapiq.approximator as approximator_module
from shapiq import ExactComputer
from shapiq.approximator import PolySHAP
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset
from shapiq_games.synthetic import DummyGame

# The sampler's border-trick and the underdefined-system check both emit UserWarnings
# that are irrelevant to most assertions here; individual tests opt back in where needed.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _singleton_prior(n: int) -> list[tuple[int, ...]]:
    """Minimal prior: empty set + all singletons (the KernelSHAP frontier)."""
    return list(powerset(range(n), max_size=1))


def _kadd_frontier_size(n: int, max_order: int) -> int:
    """Number of subsets of ``range(n)`` of size ``0 .. max_order``."""
    return int(sum(binom(n, k) for k in range(max_order + 1)))


@cache
def _exact_sv(n: int, interaction: tuple[int, ...]) -> tuple[float, ...]:
    """Ground-truth Shapley values for ``DummyGame(n, interaction)`` via ExactComputer."""
    game = DummyGame(n, interaction)
    values = ExactComputer(game, n_players=n)(index="SV", order=1).get_n_order_values(1)
    return tuple(float(v) for v in np.asarray(values))


def _estimate_vector(approx: PolySHAP, budget: int, game: DummyGame) -> np.ndarray:
    """Run ``approx`` and return its per-player Shapley-value vector of length ``n``."""
    sv = approx.approximate(budget, game)
    return np.array([sv[(i,)] for i in range(approx.n)])


def _generate_seeds(rng_seed: int = 20260617, count: int = 50) -> list[int]:
    """Generate a fixed, reproducible list of seeds spanning a wide value range."""
    return [int(s) for s in np.random.default_rng(rng_seed).integers(0, 2**31 - 1, size=count)]


SEEDS: list[int] = _generate_seeds()

# Shared configuration for the seeded sweeps.
_SEEDED_N = 12
_SEEDED_INTERACTION = (1, 2)
_SEEDED_MAX_TERMS = 30
_SEEDED_BUDGET = 400

# The three modes, as (label, constructor-kwargs) pairs, reused by shared tests.
_MODES: list[tuple[str, dict]] = [
    ("kadd_order1", {"max_order": 1}),
    ("kadd_order2", {"max_order": 2}),
    ("partial", {"max_terms": 12}),
    ("prior", {"prior_frontier": _singleton_prior(7)}),
]
_MODE_IDS = [label for label, _ in _MODES]
_MODE_KWARGS = [kwargs for _, kwargs in _MODES]


# ---------------------------------------------------------------------------
# Registration / exports
# ---------------------------------------------------------------------------


def test_polyshap_is_exported_and_registered():
    """PolySHAP must be publicly exported and registered as an SV approximator."""
    assert "PolySHAP" in approximator_module.__all__
    assert PolySHAP in approximator_module.SV_APPROXIMATORS


# ---------------------------------------------------------------------------
# Frontier construction: k-additive mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "max_order"), [(3, 1), (3, 2), (7, 1), (7, 2), (10, 3)])
def test_kadd_frontier_size_and_attributes(n, max_order):
    """The k-additive frontier holds every subset up to ``max_order`` and sets attributes."""
    approx = PolySHAP(n, max_order=max_order)
    expected_size = _kadd_frontier_size(n, max_order)
    assert approx.n == n
    assert approx.max_order == 1  # the *output* order: PolySHAP always targets SVs
    assert approx.min_order == 0
    assert approx.top_order is False
    assert approx.iteration_cost == 1
    assert approx.index == "SV"
    assert len(approx.explanation_frontier) == expected_size
    assert approx.n_variables == expected_size - 1


def test_default_constructor_is_two_additive():
    """``PolySHAP(n)`` defaults to the 2-additive frontier."""
    n = 6
    approx = PolySHAP(n)
    assert len(approx.explanation_frontier) == _kadd_frontier_size(n, 2)


def test_kadd_order1_frontier_is_kernelshap_frontier():
    """``max_order=1`` yields exactly the empty set plus all singletons."""
    n = 6
    frontier = PolySHAP(n, max_order=1).explanation_frontier
    assert len(frontier) == n + 1
    assert max(len(S) for S in frontier) == 1


@pytest.mark.parametrize("n", [4, 7])
def test_kadd_frontier_contains_empty_set_and_all_singletons(n):
    """The empty set sits at index 0 and every singleton is present."""
    frontier = PolySHAP(n, max_order=2).explanation_frontier
    assert frontier[()] == 0
    for i in range(n):
        assert (i,) in frontier


@pytest.mark.parametrize("excluded_size", [2, 3])
def test_kadd_sizes_to_exclude_removes_those_sizes(excluded_size):
    """Excluded higher-order sizes are absent, while singletons are always kept."""
    n = 6
    approx = PolySHAP(n, max_order=4, sizes_to_exclude={excluded_size})
    for coalition in approx.explanation_frontier:
        assert len(coalition) != excluded_size
    for i in range(n):
        assert (i,) in approx.explanation_frontier


def test_kadd_sizes_to_exclude_singletons_is_ignored():
    """Singletons are mandatory, so excluding size 1 has no effect (no error)."""
    n = 4
    approx = PolySHAP(n, max_order=2, sizes_to_exclude={1})
    for i in range(n):
        assert (i,) in approx.explanation_frontier


def test_kadd_frontier_positions_are_unique():
    """Each frontier term maps to a unique, contiguous column position."""
    approx = PolySHAP(5, max_order=2)
    positions = sorted(approx.explanation_frontier.values())
    assert positions == list(range(len(positions)))


def test_interaction_matrix_binary_shape():
    """``interaction_matrix_binary`` has shape ``(|frontier|, n)``."""
    n = 4
    approx = PolySHAP(n, max_order=2)
    assert approx.interaction_matrix_binary.shape == (len(approx.explanation_frontier), n)


# ---------------------------------------------------------------------------
# Frontier construction: partial mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "max_terms"),
    [
        (7, 8),  # empty + singletons only, no higher-order extension
        (7, 12),  # empty + singletons + 4 pairs
        (7, 25),  # empty + singletons + 17 pairs (21 pairs exist -> fits)
    ],
)
def test_partial_frontier_size_when_within_capacity(n, max_terms):
    """When capacity allows, the partial frontier holds exactly ``max_terms`` entries."""
    approx = PolySHAP(n, max_terms=max_terms, random_state=42)
    assert len(approx.explanation_frontier) == max_terms
    assert approx.n_variables == max_terms - 1


def test_partial_frontier_capped_by_capacity_when_candidates_exhausted():
    """If ``max_terms`` exceeds the available terms up to ``max_order`` the frontier caps out."""
    n = 5  # capacity at order 2 is 1 + 5 + C(5, 2) = 16
    approx = PolySHAP(n, max_order=2, max_terms=100, random_state=0)
    assert len(approx.explanation_frontier) == _kadd_frontier_size(n, 2)
    assert max(len(S) for S in approx.explanation_frontier) == 2


def test_partial_is_bounded_by_max_order():
    """Partial extension never adds terms above ``max_order``."""
    order2 = PolySHAP(6, max_order=2, max_terms=40, random_state=0)
    assert max(len(S) for S in order2.explanation_frontier) == 2
    order3 = PolySHAP(6, max_order=3, max_terms=40, random_state=0)
    assert max(len(S) for S in order3.explanation_frontier) == 3


def test_partial_always_contains_all_singletons():
    """Every singleton appears regardless of the cap."""
    n = 7
    approx = PolySHAP(n, max_order=3, max_terms=15, random_state=1)
    for i in range(n):
        assert (i,) in approx.explanation_frontier


def test_partial_sizes_to_exclude_omits_those_sizes():
    """Excluded sizes never appear among the randomly selected terms."""
    approx = PolySHAP(7, max_order=3, max_terms=25, sizes_to_exclude={2}, random_state=0)
    for coalition in approx.explanation_frontier:
        assert len(coalition) != 2


def test_partial_fills_whole_low_orders_before_sampling_boundary_order():
    """Partial mode realizes the paper's ``I_ell``: complete low orders, partial top order.

    With ``n=6``, ``max_order=3`` and ``max_terms=30`` the frontier holds the empty set,
    6 singletons and all 15 pairs (22 terms), then a *random* subset of the 20 triples to
    reach 30. Every pair must be present while only some triples are.
    """
    n, max_terms = 6, 30
    frontier = PolySHAP(n, max_order=3, max_terms=max_terms, random_state=0).explanation_frontier
    all_pairs = list(powerset(range(n), min_size=2, max_size=2))
    triples = [S for S in frontier if len(S) == 3]

    assert all(pair in frontier for pair in all_pairs)  # every lower-order term kept
    assert 0 < len(triples) < int(binom(n, 3))  # boundary order only partially included
    assert len(frontier) == max_terms


def test_partial_different_seeds_produce_different_frontiers():
    """Different random states select different higher-order interactions."""
    n, max_terms = 7, 20
    a1 = PolySHAP(n, max_order=3, max_terms=max_terms, random_state=1)
    a2 = PolySHAP(n, max_order=3, max_terms=max_terms, random_state=2)
    assert set(a1.explanation_frontier) != set(a2.explanation_frontier)


def test_partial_same_seed_reproduces_frontier():
    """The same random state yields an identical frontier."""
    n, max_terms = 7, 20
    a1 = PolySHAP(n, max_order=3, max_terms=max_terms, random_state=7)
    a2 = PolySHAP(n, max_order=3, max_terms=max_terms, random_state=7)
    assert set(a1.explanation_frontier) == set(a2.explanation_frontier)


# ---------------------------------------------------------------------------
# Frontier construction: prior mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 7, 10])
def test_prior_frontier_matches_supplied_ordering(n):
    """The prior frontier equals ``prior_frontier`` with positions in enumeration order."""
    prior = [*_singleton_prior(n), (0, 1), (1, 2)]
    approx = PolySHAP(n, prior_frontier=prior)
    assert len(approx.explanation_frontier) == len(prior)
    assert approx.n_variables == len(prior) - 1
    for pos, coalition in enumerate(prior):
        assert approx.explanation_frontier[coalition] == pos


def test_informed_prior_recovers_exact_interaction_game():
    """A prior carrying the game's true interaction recovers the exact Shapley values.

    This is the realistic use of *prior* mode: on a 2-additive game whose only
    interaction is the pair ``(1, 2)``, supplying that pair (plus the singletons) as the
    frontier lets PolySHAP recover the exact SVs from a subsampled budget (``n=12``,
    ``budget << 2**n``), whereas a singleton-only prior (KernelSHAP) cannot.
    """
    n, budget, interaction = 12, 200, (1, 2)
    exact = np.asarray(_exact_sv(n, interaction))

    informed = [*_singleton_prior(n), interaction]
    est = _estimate_vector(
        PolySHAP(n, prior_frontier=informed, random_state=1), budget, DummyGame(n, interaction)
    )
    np.testing.assert_allclose(est, exact, atol=1e-6)

    # A singleton-only prior lacks the interaction term and leaves a visible error.
    est_singletons = _estimate_vector(
        PolySHAP(n, prior_frontier=_singleton_prior(n), random_state=1),
        budget,
        DummyGame(n, interaction),
    )
    assert np.sqrt(np.mean((est_singletons - exact) ** 2)) > 1e-3


# ---------------------------------------------------------------------------
# Frontier construction: validation / error paths
# ---------------------------------------------------------------------------


def test_prior_frontier_with_max_order_raises():
    """``prior_frontier`` combined with ``max_order`` is rejected."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        PolySHAP(5, prior_frontier=_singleton_prior(5), max_order=2)


def test_prior_frontier_with_max_terms_raises():
    """``prior_frontier`` combined with ``max_terms`` is rejected."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        PolySHAP(5, prior_frontier=_singleton_prior(5), max_terms=10)


def test_prior_frontier_with_sizes_to_exclude_raises():
    """``prior_frontier`` combined with ``sizes_to_exclude`` is rejected."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        PolySHAP(5, prior_frontier=_singleton_prior(5), sizes_to_exclude={2})


def test_max_terms_below_minimum_raises():
    """``max_terms`` must leave room for the empty set and all singletons."""
    n = 6
    with pytest.raises(ValueError, match="at least"):
        PolySHAP(n, max_terms=n)  # needs >= n + 1


def test_prior_missing_singleton_raises():
    """A prior that omits a singleton is rejected by the base-class check."""
    n = 5
    incomplete = [(), (0,), (1,), (2,)]  # players 3 and 4 absent
    with pytest.raises(ValueError, match="main effects"):
        PolySHAP(n, prior_frontier=incomplete)


# ---------------------------------------------------------------------------
# Approximation output contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", _MODE_KWARGS, ids=_MODE_IDS)
def test_output_is_valid_interaction_values(kwargs):
    """Every mode returns a well-formed SV ``InteractionValues`` object."""
    n, budget = 7, 200
    game = DummyGame(n, (1, 2))
    sv = PolySHAP(n, **kwargs, random_state=0).approximate(budget, game)
    assert isinstance(sv, InteractionValues)
    assert sv.index == "SV"
    assert sv.max_order == 1
    assert sv.min_order == 0
    assert sv.estimation_budget <= budget
    for i in range(n):
        assert (i,) in sv.interaction_lookup


@pytest.mark.parametrize("kwargs", _MODE_KWARGS, ids=_MODE_IDS)
def test_budget_never_exceeded(kwargs):
    """The game is never queried more than ``budget`` times."""
    n, budget = 7, 150
    game = DummyGame(n, (1, 2))
    PolySHAP(n, **kwargs, random_state=0).approximate(budget, game)
    assert game.access_counter <= budget


@pytest.mark.parametrize("kwargs", _MODE_KWARGS, ids=_MODE_IDS)
def test_sv_efficiency_holds(kwargs):
    """The Shapley values sum to ``v(N) - v(empty) = 2.0`` for ``DummyGame(7, (1, 2))``."""
    n = 7
    game = DummyGame(n, (1, 2))
    sv = PolySHAP(n, **kwargs, random_state=0).approximate(500, game)
    assert np.sum(sv.values) == pytest.approx(2.0, abs=0.05)


def test_pairing_trick_produces_valid_output():
    """``pairing_trick=True`` still yields accurate estimates (order-1 path)."""
    n, budget = 7, 200
    game = DummyGame(n, (1, 2))
    est = _estimate_vector(
        PolySHAP(n, max_order=1, pairing_trick=True, random_state=7), budget, game
    )
    exact = np.asarray(_exact_sv(n, (1, 2)))
    np.testing.assert_allclose(est, exact, atol=0.15)


# ---------------------------------------------------------------------------
# Convergence to the ExactComputer ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", _MODE_KWARGS, ids=_MODE_IDS)
def test_converges_to_exact_sv_with_sufficient_budget(kwargs):
    """With ample budget every mode recovers the exact Shapley values (n <= 10)."""
    n, budget = 7, 600
    game = DummyGame(n, (1, 2))
    est = _estimate_vector(PolySHAP(n, **kwargs, random_state=42), budget, game)
    exact = np.asarray(_exact_sv(n, (1, 2)))
    np.testing.assert_allclose(est, exact, atol=0.1)


def test_matching_frontier_order_recovers_exact_two_additive_game():
    """A 2-additive game is recovered exactly by ``max_order=2`` but not ``max_order=1``.

    Uses ``n = 12`` with ``budget << 2**n`` so the estimate is genuinely subsampled
    rather than a full-space enumeration.
    """
    n, budget = 12, 200
    interaction = (1, 2)
    exact = np.asarray(_exact_sv(n, interaction))

    est_order2 = _estimate_vector(
        PolySHAP(n, max_order=2, random_state=1), budget, DummyGame(n, interaction)
    )
    np.testing.assert_allclose(est_order2, exact, atol=1e-6)

    est_order1 = _estimate_vector(
        PolySHAP(n, max_order=1, random_state=1), budget, DummyGame(n, interaction)
    )
    assert np.sqrt(np.mean((est_order1 - exact) ** 2)) > 1e-3


def test_higher_order_reduces_error_on_three_additive_game():
    """On a 3-way interaction game, ``max_order=3`` recovers exact SVs while order 2 lags."""
    n, budget = 12, 300
    interaction = (1, 2, 3)
    exact = np.asarray(_exact_sv(n, interaction))

    rmse = {}
    for order in (2, 3):
        est = _estimate_vector(
            PolySHAP(n, max_order=order, random_state=1), budget, DummyGame(n, interaction)
        )
        rmse[order] = float(np.sqrt(np.mean((est - exact) ** 2)))

    np.testing.assert_allclose(
        _estimate_vector(
            PolySHAP(n, max_order=3, random_state=1), budget, DummyGame(n, interaction)
        ),
        exact,
        atol=1e-6,
    )
    assert rmse[3] < rmse[2]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3])
def test_tiny_games(n):
    """PolySHAP works for very small player counts and stays efficient."""
    game = DummyGame(n, (0, 1))
    sv = PolySHAP(n, max_order=1, random_state=0).approximate(2**n, game)
    assert np.sum(sv.values) == pytest.approx(2.0, abs=1e-6)


def test_budget_smaller_than_frontier_still_runs():
    """A budget far below the frontier size runs and returns a full SV vector."""
    n = 8
    game = DummyGame(n, (1, 2))
    sv = PolySHAP(n, max_order=2, random_state=0).approximate(6, game)
    assert isinstance(sv, InteractionValues)
    for i in range(n):
        assert (i,) in sv.interaction_lookup


def test_underdefined_system_warns():
    """An under-sampled least-squares system emits an explanatory UserWarning."""
    n = 8  # order-2 frontier has 36 variables; budget of 10 is far too small
    game = DummyGame(n, (1, 2))
    with pytest.warns(UserWarning, match="underdefined"):
        PolySHAP(n, max_order=2, random_state=0).approximate(10, game)


def test_estimated_flag_true_when_budget_small():
    """``estimated`` is True when the budget is below the full coalition space."""
    n = 7
    sv = PolySHAP(n, max_order=1, random_state=0).approximate(50, DummyGame(n, (1, 2)))
    assert sv.estimated is True


def test_estimated_flag_false_when_full_space_covered():
    """``estimated`` is False when the budget exceeds ``2**n``."""
    n = 5
    sv = PolySHAP(n, max_order=1, random_state=0).approximate(2**n + 10, DummyGame(n, (1, 2)))
    assert sv.estimated is False


# ---------------------------------------------------------------------------
# Reproducibility under a fixed random_state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", _MODE_KWARGS, ids=_MODE_IDS)
def test_random_state_reproducibility(kwargs):
    """Identical ``random_state`` yields bit-identical estimates for every mode."""
    n, budget = 7, 200
    sv1 = PolySHAP(n, **kwargs, random_state=99).approximate(budget, DummyGame(n, (1, 2)))
    sv2 = PolySHAP(n, **kwargs, random_state=99).approximate(budget, DummyGame(n, (1, 2)))
    np.testing.assert_array_equal(sv1.values, sv2.values)


def test_prior_singleton_matches_kadd_order1():
    """A singleton prior and ``max_order=1`` share the same frontier and results."""
    n, budget, seed = 7, 200, 5
    sv_prior = PolySHAP(n, prior_frontier=_singleton_prior(n), random_state=seed).approximate(
        budget, DummyGame(n, (1, 2))
    )
    sv_kadd = PolySHAP(n, max_order=1, random_state=seed).approximate(budget, DummyGame(n, (1, 2)))
    np.testing.assert_array_almost_equal(sv_prior.values, sv_kadd.values, decimal=12)


# ---------------------------------------------------------------------------
# Seeded sweeps over 50 fixed seeds
# ---------------------------------------------------------------------------


def test_seed_list_is_deterministic_and_unique():
    """The fixed seed list is regenerable and contains 50 unique seeds."""
    assert _generate_seeds() == SEEDS
    assert len(SEEDS) == 50
    assert len(set(SEEDS)) == 50


def test_seeded_runs_are_reproducible():
    """For every seed, two independent partial-mode runs are bit-identical."""
    for seed in SEEDS:
        est1 = _estimate_vector(
            PolySHAP(_SEEDED_N, max_terms=_SEEDED_MAX_TERMS, random_state=seed),
            _SEEDED_BUDGET,
            DummyGame(_SEEDED_N, _SEEDED_INTERACTION),
        )
        est2 = _estimate_vector(
            PolySHAP(_SEEDED_N, max_terms=_SEEDED_MAX_TERMS, random_state=seed),
            _SEEDED_BUDGET,
            DummyGame(_SEEDED_N, _SEEDED_INTERACTION),
        )
        np.testing.assert_array_equal(
            est1, est2, err_msg=f"non-reproducible result for seed {seed}"
        )


def test_seeded_estimates_are_unbiased_with_low_variance():
    """Across 50 seeds the partial-mode estimates cluster tightly around the true SVs."""
    exact = np.asarray(_exact_sv(_SEEDED_N, _SEEDED_INTERACTION))

    estimates = np.empty((len(SEEDS), _SEEDED_N))
    for row, seed in enumerate(SEEDS):
        estimates[row] = _estimate_vector(
            PolySHAP(_SEEDED_N, max_terms=_SEEDED_MAX_TERMS, random_state=seed),
            _SEEDED_BUDGET,
            DummyGame(_SEEDED_N, _SEEDED_INTERACTION),
        )

    # Near-unbiased: the mean over seeds matches the exact Shapley values.
    np.testing.assert_allclose(estimates.mean(axis=0), exact, atol=0.02)
    # Tight spread around the truth.
    assert estimates.std(axis=0).max() < 0.05
    # Efficiency holds on average.
    assert estimates.sum(axis=1).mean() == pytest.approx(2.0, abs=0.01)


def test_error_decreases_as_budget_grows():
    """Mean RMSE over 50 seeds drops monotonically as the budget increases (order-1)."""
    n, interaction = 8, (1, 2)
    exact = np.asarray(_exact_sv(n, interaction))
    budgets = [40, 80, 160, 240]  # all below 2**8 = 256, so genuinely estimated

    mean_rmse = []
    for budget in budgets:
        rmses = [
            np.sqrt(
                np.mean(
                    (
                        _estimate_vector(
                            PolySHAP(n, max_order=1, random_state=seed),
                            budget,
                            DummyGame(n, interaction),
                        )
                        - exact
                    )
                    ** 2
                )
            )
            for seed in SEEDS
        ]
        mean_rmse.append(float(np.mean(rmses)))

    assert all(later < earlier for earlier, later in pairwise(mean_rmse))
