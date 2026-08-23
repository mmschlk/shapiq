"""This test module contains all tests regarding the LeverageSHAP regression approximator."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter

import numpy as np
import pytest

import shapiq.approximator.regression.leverageshap as leverageshap_module
from shapiq.approximator.regression import KernelSHAP, LeverageSHAP
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import SOUM, DummyGame

# Random but pre-defined seeds to ensure reproducibility but prevent overfitting to a single seed choice
DIVERSE_SEEDS = [
    0,
    42,
    1337,
    9999,
    12345,
    23096863,
    1589215,
    240926,
    12358259,
    4236902346,
    633126624,
    436135,
    5342326142,
    46233152,
    325235,
]


@pytest.mark.parametrize("n", [3, 7, 10])
def test_initialization(n):
    """Tests the initialization of the LeverageSHAP approximator."""
    approximator = LeverageSHAP(n)
    assert approximator.n == n
    assert approximator.max_order == 1
    assert approximator.top_order is False
    assert approximator.min_order == 0
    assert approximator.index == "SV"


@pytest.mark.parametrize(("n", "budget"), [(7, 380), (7, 100)])
@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_approximate(n, budget, seed):
    """Tests the approximation of the LeverageSHAP approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    approximator = LeverageSHAP(n, random_state=seed)
    sv_estimates = approximator.approximate(budget, game)

    assert isinstance(sv_estimates, InteractionValues)
    assert sv_estimates.max_order == 1
    assert sv_estimates.min_order == 0
    assert sv_estimates.index == "SV"
    # estimation_budget reports the realized number of coalitions evaluated. With the
    # default deterministic_counts=True this is exact (see test_estimation_budget_matches_
    # exact_formula below); it must equal the game's access count and never exceed full
    # enumeration (2**n).
    assert sv_estimates.estimation_budget == game.access_counter
    assert sv_estimates.estimation_budget <= 2**n
    assert sv_estimates.estimated != (budget >= 2**n)

    # The access counter should be at most 2**n, since LeverageSHAP caps the budget at 2**n and does not make redundant calls.
    assert game.access_counter <= 2**n

    # players 1 and 2 should be the most important (DummyGame interaction on (1, 2))
    assert sv_estimates[(1,)] == pytest.approx(0.6429, abs=0.15)
    assert sv_estimates[(2,)] == pytest.approx(0.6429, abs=0.15)

    # efficiency axiom: sum of SVs == v(N) - v({})
    efficiency = np.sum(sv_estimates.values[1:])
    v_grand = game(np.ones((1, n), dtype=bool))[0]
    v_empty = game(np.zeros((1, n), dtype=bool))[0]
    assert efficiency == pytest.approx(v_grand - v_empty, abs=1e-6)


def test_exact_matches_exactcomputer_on_small_game():
    """With a full budget, LeverageSHAP should match ExactComputer on a small game."""
    n = 6
    budget = 2**n
    game = DummyGame(n, interaction=(0, 2))

    # Use ExactComputer here as the ground-truth reference.
    # The point of the test is to check that LeverageSHAP reaches the exact answer
    # once the budget covers the full coalition space.
    exact = ExactComputer(game, n)
    exact_sv = exact("SV")

    approximator = LeverageSHAP(n, random_state=42)
    result = approximator.approximate(budget=budget, game=game)

    # NumPy arrays need elementwise comparison, and floating-point code may differ by
    # tiny rounding noise, so assert_allclose is the right check here.
    # The results below should be exactly equal, but we allow for a tiny absolute tolerance
    # to account for any minor floating-point discrepancies that may arise from different
    # computational paths in the two methods.
    np.testing.assert_allclose(result.values, exact_sv.values, atol=1e-8, rtol=0.0)

    # With a full budget, the approximation is exact, so estimated should be False. The parameter 'estimated' indicates whether the result is an estimate (True) or exact (False). Since we are using a full budget that covers all coalitions, LeverageSHAP should be able to compute the exact Shapley values, and thus estimated should be False.
    assert result.estimated is False
    assert (
        result.estimation_budget == budget
    )  # The estimation budget should match the full budget used for approximation.


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_tiny_n_budget_two_symmetric_game(seed):
    """Tiny-n edge case: with n=2 and the minimum valid budget, the solver should still work."""
    n = 2

    def symmetric_game(Z):
        return Z.astype(float).sum(axis=1)

    # This is the smallest valid setting for LeverageSHAP.
    # It checks that the solver still returns a sensible result when there are no
    # interior coalition sizes to learn from.
    approximator = LeverageSHAP(n, random_state=seed)
    result = approximator.approximate(budget=2, game=symmetric_game)

    # The game is perfectly symmetric, so the two players should receive the same SV.
    assert result.estimated is True
    np.testing.assert_allclose(result.values[1:], np.array([1.0, 1.0]), atol=1e-12, rtol=0.0)
    assert result.values[1:].sum() == pytest.approx(2.0, abs=1e-12)


def test_exact_recovery_additive_game():
    """With budget == 2^n, LeverageSHAP should recover exact SVs on an additive game."""
    n = 5
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def additive_game(Z):
        return Z.astype(float) @ weights

    approximator = LeverageSHAP(n, random_state=0)
    result = approximator.approximate(budget=2**n, game=additive_game)

    assert result.values[1:] == pytest.approx(weights, abs=1e-8)
    assert result.estimated is False


def test_budget_too_small_raises():
    """Budget < 2 should raise a ValueError."""
    approximator = LeverageSHAP(n=5)
    with pytest.raises(ValueError, match="Budget must be at least 2"):
        approximator.approximate(budget=1, game=lambda Z: np.zeros(len(Z)))


def _independent_find_c(n: int, m: int) -> float:
    """Standalone re-derivation of Eq. 12's oversampling constant (binary search).

    Written independently of ``LeverageSHAP._find_c`` (own bracketing/bisection loop) so
    that ``test_per_size_counts_match_expected_formula_and_symmetry`` is not merely
    checking the implementation against itself.
    """
    target = m - 2
    if n < 2 or target <= 0:
        return 0.0
    binoms = [math.comb(n, s) for s in range(1, n)]
    hi = 1.0
    while sum(min(b, 2.0 * hi) for b in binoms) < target:
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum(min(b, 2.0 * mid) for b in binoms) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _independent_expected_pair_counts(n: int, budget: int) -> dict[int, int]:
    """Standalone re-derivation of the deterministic per-half-size pair counts ``m_s``.

    Mirrors the largest-remainder apportionment described in the class docstring and
    ``BRIEF.md`` item 4 (expected Binomial counts, rounded so the total matches the
    budget exactly), but is written directly against the documented formula rather than
    by calling ``LeverageSHAP._bernoulli_sample_deterministic``.
    """
    m = min(budget, 2**n)
    c = _independent_find_c(n, m)
    two_c = 2.0 * c
    target_pairs = (m - 2) // 2
    half_sizes = list(range(1, n // 2 + 1))

    m_s: dict[int, int] = {}
    frac: dict[int, float] = {}
    exhaustive: dict[int, bool] = {}
    pool_size_of: dict[int, int] = {}
    for s in half_sizes:
        is_middle = (n % 2 == 0) and (s == n // 2)
        full_count = math.comb(n, s)
        pool_size = math.comb(n - 1, s - 1) if is_middle else full_count
        pool_size_of[s] = pool_size
        if full_count <= two_c:
            m_s[s] = pool_size
            frac[s] = 0.0
            exhaustive[s] = True
        else:
            mu = two_c * s / n if is_middle else two_c
            floor_mu = int(mu)
            m_s[s] = min(floor_mu, pool_size)
            frac[s] = mu - floor_mu
            exhaustive[s] = False

    shortfall = target_pairs - sum(m_s.values())
    for s in sorted(half_sizes, key=lambda s: -frac[s]):
        if shortfall <= 0:
            break
        if exhaustive[s] or m_s[s] >= pool_size_of[s]:
            continue
        m_s[s] += 1
        shortfall -= 1
    return m_s


@pytest.mark.parametrize(
    ("n", "budget"),
    [
        (4, 2),
        (4, 3),
        (4, 7),
        (4, 16),
        (4, 17),
        (4, 100),
        (5, 2),
        (5, 5),
        (5, 9),
        (5, 32),
        (5, 33),
        (6, 2),
        (6, 21),
        (6, 40),
        (6, 63),
        (6, 64),
        (6, 65),
        (6, 1000),
        (8, 3),
        (8, 255),
        (8, 256),
        (8, 257),
    ],
)
def test_estimation_budget_matches_exact_formula(n, budget):
    """With deterministic_counts=True (default), the realized budget is exact.

    Covers odd budgets and budgets exceeding 2**n, per BRIEF.md item 4: the number of
    game evaluations must equal ``2 + 2 * ((min(budget, 2**n) - 2) // 2)`` exactly, with
    no over- or undershoot.
    """
    game = DummyGame(n, interaction=(0, 1))
    result = LeverageSHAP(n, random_state=0).approximate(budget, game)
    expected = 2 + 2 * ((min(budget, 2**n) - 2) // 2)
    assert result.estimation_budget == expected
    assert result.estimation_budget == game.access_counter


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10, 11, 12])
@pytest.mark.parametrize("budget_fraction", [0.1, 0.3, 0.6])
def test_per_size_counts_match_expected_formula_and_symmetry(n, budget_fraction):
    """Per-size sampled counts must be symmetric (count(s) == count(n - s)) and match
    an independently re-derived expected-count formula (BRIEF.md item 4).
    """
    budget = max(2, int(budget_fraction * 2**n))
    approximator = LeverageSHAP(n, random_state=0)
    Z, _ = approximator._sample(budget)
    sizes = Z.sum(axis=1)
    counts = Counter(int(s) for s in sizes[2:])  # rows 0, 1 are empty/grand

    expected_pairs = _independent_expected_pair_counts(n, budget)
    for s, pairs in expected_pairs.items():
        is_middle = (n % 2 == 0) and (s == n // 2)
        if is_middle:
            assert counts.get(s, 0) == 2 * pairs
        else:
            assert counts.get(s, 0) == pairs
            assert counts.get(n - s, 0) == pairs
        # symmetry: a size and its complement are always equally represented
        assert counts.get(s, 0) == counts.get(n - s, 0)


def test_saturation_all_rows_distinct_and_exact_on_soum():
    """At budget >= 2**n, all 2**n distinct coalitions are sampled and the estimate is
    exact (not merely close) on a SOUM game (BRIEF.md item 4).
    """
    n = 6
    budget = 2**n

    approximator = LeverageSHAP(n, random_state=0)
    Z, _ = approximator._sample(budget)
    assert Z.shape[0] == 2**n
    rows = {tuple(row) for row in Z}
    assert len(rows) == 2**n

    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    exact = ExactComputer(game, n)
    exact_sv = exact("SV")

    result = LeverageSHAP(n, random_state=1).approximate(budget, game)
    assert result.estimated is False
    np.testing.assert_allclose(result.values, exact_sv.values, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_unanimity_game_exact_svs(seed):
    """Unanimity game v(S) = 1 iff T ⊆ S is fully non-additive.

    True Shapley values: 1/|T| for players in T, 0 for others.
    At full budget every interior coalition has inclusion probability 1, so the IS
    weight w(s) = (s-1)!(n-s-1)!/n! is a fixed global multiple ((n-1)x) of the Shapley
    kernel weight. A global scale does not change the weighted-least-squares argmin, so
    the per-run full-budget solution is *exact* (not merely unbiased in expectation).
    We assert the exact Shapley values, plus the efficiency and ordering properties.
    """
    n = 6
    T = frozenset({1, 3, 5})

    def unanimity_game(Z):
        return np.array([1.0 if all(row[j] for j in T) else 0.0 for row in Z])

    approximator = LeverageSHAP(n, random_state=seed)
    result = approximator.approximate(budget=2**n, game=unanimity_game)

    # exact Shapley values: 1/|T| inside T, 0 outside
    expected = np.array([1.0 / len(T) if i in T else 0.0 for i in range(n)])
    np.testing.assert_allclose(result.values[1:], expected, atol=1e-8, rtol=0.0)

    # efficiency must always hold exactly
    assert result.values[1:].sum() == pytest.approx(1.0, abs=1e-8)

    sv = result.values[1:]
    t_svs = [sv[i] for i in T]
    non_t_svs = [sv[i] for i in range(n) if i not in T]
    assert min(t_svs) > max(non_t_svs)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_large_n_efficiency_axiom(seed):
    """For large n, binom(n, s) would overflow or lose precision if not cancelled.

    Verifies that the IS weight cancellation (1/binom · binom = 1/(s(n-s))) keeps
    the efficiency axiom satisfied to machine precision even for n=20.
    """
    n = 20
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal(n)

    def additive_game(Z):
        return Z.astype(float) @ weights

    v_grand = additive_game(np.ones((1, n)))[0]
    v_empty = additive_game(np.zeros((1, n)))[0]

    approximator = LeverageSHAP(n, random_state=seed)
    result = approximator.approximate(budget=500, game=additive_game)

    assert result.values[1:].sum() == pytest.approx(v_grand - v_empty, abs=1e-6)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_skewed_interaction_game(seed):
    """Game where one dominant player multiplies everyone else's contribution by 1000x.

    This creates a highly ill-conditioned design matrix: without W^{1/2} row-scaling
    (i.e. if normal equations A^T W A were formed explicitly), the condition number
    would be squared and the solver would lose ~6 digits of precision.
    LeverageSHAP's lstsq-based WLS must still satisfy efficiency and rank player 0 highest.
    """
    # Use a moderately sized game so the approximation has enough structure to expose
    # numerical issues while still being small enough for a fast regression test.
    n = 7
    # The first player is the special one whose presence changes the payoff scale.
    dominant = 0

    def skewed_game(Z):
        # Give every coalition containing the dominant player a huge payoff multiplier,
        # and every coalition without it a tiny multiplier, to create extreme imbalance.
        scale = np.where(Z[:, dominant], 1000.0, 0.001)
        # Sum the contributions of all non-dominant players for each coalition row.
        other_sum = Z[:, 1:].astype(float).sum(axis=1)
        # Return the scaled coalition value that depends strongly on whether player 0 is present.
        return scale * other_sum

    # Compute the grand-coalition value, which is the reference value for the efficiency check.
    v_grand = skewed_game(np.ones((1, n)))[0]
    # The empty coalition has no included players, so its value is set to zero explicitly.
    v_empty = 0.0

    # Build the LeverageSHAP approximator with a fixed seed so the test is deterministic.
    approximator = LeverageSHAP(n, random_state=seed)
    # Approximate Shapley values under the skewed payoff function using the chosen budget.
    result = approximator.approximate(budget=300, game=skewed_game)

    # efficiency is baked in algebraically via the efficiency_shift construction, so it must
    # hold to near machine precision even under extreme scale differences.
    assert result.values[1:].sum() == pytest.approx(v_grand - v_empty, rel=1e-10)

    # dominant player (index 0) must have the highest SV
    assert result[(dominant,)] == pytest.approx(max(result.values[1:]), abs=1e-6)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_reproducibility(seed):
    """Same seed should produce identical approximations across runs."""
    n, budget = 6, 20

    # Use separate game instances so access counters don't interfere
    game1 = DummyGame(n, interaction=(1, 2))
    game2 = DummyGame(n, interaction=(1, 2))

    # Run 1
    approx1 = LeverageSHAP(n, random_state=seed)
    res1 = approx1.approximate(budget, game1)

    # Run 2
    approx2 = LeverageSHAP(n, random_state=seed)
    res2 = approx2.approximate(budget, game2)

    # Values should be identical
    np.testing.assert_array_equal(res1.values, res2.values)

    # Other run metadata should match exactly
    assert res1.estimation_budget == res2.estimation_budget
    assert res1.estimated == res2.estimated


def test_different_seeds_draw_different_rows():
    """Different seeds must (usually) draw a different set of sampled coalitions.

    Complements test_reproducibility (same seed -> identical .values): here we check the
    row level directly on a SOUM game, since two different row draws could in principle
    still yield identical Shapley-value estimates by coincidence.
    """
    n, budget = 6, 20  # non-exhaustive: budget < 2**n == 64

    Z_a, _ = LeverageSHAP(n, random_state=0)._sample(budget)
    Z_b, _ = LeverageSHAP(n, random_state=1)._sample(budget)

    rows_a = {tuple(row) for row in Z_a.astype(bool)}
    rows_b = {tuple(row) for row in Z_b.astype(bool)}
    assert rows_a != rows_b


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_deterministic_counts_false_smoke(seed):
    """``deterministic_counts=False`` (the literal Binomial Algorithm 2) must run,
    report ``estimated``, draw distinct rows, and be seed-reproducible (BRIEF.md item 4).
    """
    n, budget = 6, 20

    game1 = DummyGame(n, interaction=(1, 2))
    game2 = DummyGame(n, interaction=(1, 2))

    approx1 = LeverageSHAP(n, deterministic_counts=False, random_state=seed)
    res1 = approx1.approximate(budget, game1)
    approx2 = LeverageSHAP(n, deterministic_counts=False, random_state=seed)
    res2 = approx2.approximate(budget, game2)

    assert isinstance(res1, InteractionValues)
    assert res1.estimated is True
    assert res1.estimation_budget == game1.access_counter
    assert res1.estimation_budget <= 2**n

    Z, _ = LeverageSHAP(n, deterministic_counts=False, random_state=seed)._sample(budget)
    rows = [tuple(row) for row in Z.astype(bool)]
    assert len(rows) == len(set(rows)), "Binomial-path rows must be distinct"

    # Seed-reproducible: identical seed -> identical output.
    np.testing.assert_array_equal(res1.values, res2.values)
    assert res1.estimation_budget == res2.estimation_budget


def test_exact_regime_seed_independence():
    """When the budget covers the full coalition space, results must be seed-independent.

    At full budget (2**n) every coalition size has inclusion probability that rounds to
    ~1.0, so BernoulliSample draws the entire coalition space regardless of the random
    seed (there is no separate deterministic branch — the sampling probabilities simply
    saturate). Two different seeds must therefore yield identical output. This test
    asserts that behavior.
    """
    n = 6
    budget = 2**n

    game_a = DummyGame(n, interaction=(1, 2))
    game_b = DummyGame(n, interaction=(1, 2))

    # Use two different seeds to ensure seed has no effect in exact regime
    res_a = LeverageSHAP(n, random_state=0).approximate(budget, game_a)
    res_b = LeverageSHAP(n, random_state=1).approximate(budget, game_b)

    # Exact regime: outputs must be identical (bitwise for the arrays)
    np.testing.assert_array_equal(res_a.values, res_b.values)
    assert res_a.estimation_budget == res_b.estimation_budget
    assert res_a.estimated == res_b.estimated


@pytest.mark.parametrize("deterministic_counts", [True, False])
def test_stochastic_regime_seed_variability(deterministic_counts):
    """In the sampling regime, different seeds should usually produce different estimates.

    This test is conservative and robust: it runs multiple seeds and asserts that at
    least one pair of resulting value vectors differs by more than a small numerical
    tolerance. We avoid asserting that *all* seeds must differ because low budgets can
    coincidentally yield identical samples; instead we require that variability is
    observable across several independent seeds.

    Uses a SOUM game (several random basis games, n=6) rather than a DummyGame: with
    deterministic_counts=True (the default) the per-size counts are seed-independent, so
    the only source of variability across seeds is *which* rows get drawn within each
    size, and the paired WLS estimator can land on the exact answer for many seeds on a
    degenerate two-interaction game (masking variability); SOUM's richer payoff
    structure makes different row draws produce visibly different estimates instead.
    """
    n = 6
    budget = 20  # ensure budget < 2**n so sampling occurs

    def make_game():
        return SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=1)

    results = [
        LeverageSHAP(n, random_state=s, deterministic_counts=deterministic_counts)
        .approximate(budget, make_game())
        .values
        for s in DIVERSE_SEEDS
    ]

    atol = 1e-8
    found_diff = False
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if not np.allclose(results[i], results[j], atol=atol, rtol=0.0):
                found_diff = True
                break
        if found_diff:
            break

    assert found_diff, "No observable variability between seeds in stochastic regime"


def test_empirical_convergence_rate():
    """The approximation error (w.r.t. ExactComputer) should decrease when the budget increases.

    Use averaging across a few seeds to reduce stochastic noise in the test.
    """

    n = 8

    def game_factory():
        return DummyGame(n, interaction=(0, 2))

    # ground truth (ExactComputer expects (game, n_players))
    exact = ExactComputer(game_factory(), n)
    exact_sv = exact("SV").values[1:]

    def mean_error(budget: int) -> float:
        errs = []
        for s in DIVERSE_SEEDS:
            res = LeverageSHAP(n, random_state=s).approximate(budget, game_factory())
            errs.append(np.linalg.norm(exact_sv - res.values[1:]))
        return float(np.mean(errs))

    err_small = mean_error(12)
    err_medium = mean_error(24)
    err_large = mean_error(48)

    assert err_large < err_medium < err_small


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_paired_sampling_invariant(seed):
    """With ``pairing_trick=True`` (the default), every sampled coalition must appear
    together with its complement -- Algorithm 1's ``(z, z̄)`` design.

    We verify the structural invariant directly: for the empty/grand pair and every
    interior coalition in the sampled matrix, its bitwise complement is also present.
    """
    n, budget = 6, 40
    approximator = LeverageSHAP(n, pairing_trick=True, random_state=seed)
    Z, _ = approximator._sample(budget)

    # Represent each coalition as an immutable tuple so we can test set membership.
    rows = {tuple(row) for row in Z.astype(bool)}
    for row in Z.astype(bool):
        complement = tuple(~row)
        assert complement in rows, "Sampled coalition is missing its complement"


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_unpaired_sampling_same_counts_no_forced_complement(seed):
    """With ``pairing_trick=False``, per-size counts match the paired mode, rows are
    drawn without replacement (no duplicates), and -- for a non-exhaustive budget -- at
    least one row's complement is absent (BRIEF.md item 4, the "without paired sampling"
    ablation).
    """
    n, budget = 6, 40  # budget < 2**n == 64, so sampling is non-exhaustive

    Z_paired, _ = LeverageSHAP(n, pairing_trick=True, random_state=seed)._sample(budget)
    Z_unpaired, _ = LeverageSHAP(n, pairing_trick=False, random_state=seed)._sample(budget)

    # Per-size allocation is identical between modes; only which rows are drawn differs.
    counts_paired = Counter(int(s) for s in Z_paired.sum(axis=1))
    counts_unpaired = Counter(int(s) for s in Z_unpaired.sum(axis=1))
    assert counts_paired == counts_unpaired

    # No duplicate rows (without-replacement sampling in both modes).
    rows_unpaired = [tuple(row) for row in Z_unpaired.astype(bool)]
    assert len(rows_unpaired) == len(set(rows_unpaired))

    # At least one sampled coalition's complement is absent: pairing is not forced.
    rows_set = set(rows_unpaired)
    missing_complement = any(tuple(~np.array(row)) not in rows_set for row in rows_unpaired)
    assert missing_complement, "Expected at least one row without its complement present"


def test_pairing_modes_both_exact_at_full_budget():
    """Both pairing modes must reach the exact Shapley values at budget == 2**n."""
    n = 6
    budget = 2**n
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    exact = ExactComputer(game, n)
    exact_sv = exact("SV")

    for pairing_trick in (True, False):
        result = LeverageSHAP(n, pairing_trick=pairing_trick, random_state=0).approximate(
            budget, game
        )
        assert result.estimated is False
        np.testing.assert_allclose(result.values, exact_sv.values, atol=1e-10, rtol=0.0)


def test_leverageshap_vs_kernelshap_mean_error():
    """LeverageSHAP should have no larger mean error than KernelSHAP on average.

    We compare average L2 error (w.r.t. ExactComputer) across several seeds for
    both methods at the same budget. The test is conservative: we assert that
    LeverageSHAP's mean error is not greater than KernelSHAP's mean error.
    """
    n = 6
    budget = 40
    exact = ExactComputer(DummyGame(n, interaction=(0, 1)), n)
    exact_sv = exact("SV").values[1:]

    def mean_err(approximator_cls):
        errs = []
        for s in DIVERSE_SEEDS:
            approx = approximator_cls(n, random_state=s)
            res = approx.approximate(budget, DummyGame(n, interaction=(0, 1)))
            errs.append(np.linalg.norm(exact_sv - res.values[1:]))
        return float(np.mean(errs))

    err_leverage = mean_err(LeverageSHAP)
    err_kernel = mean_err(KernelSHAP)

    assert err_leverage <= err_kernel


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_exact_matches_multiple_small_games(seed):
    """Verify exact-match property on several small games and n values.

    Ensures that when budget==2**n LeverageSHAP matches ExactComputer for
    multiple small n and for both a DummyGame and an additive game.
    """
    for n in (3, 4, 5, 6):
        # DummyGame
        game1 = DummyGame(n, interaction=(0, 1))
        exact1 = ExactComputer(game1, n)
        exact_sv1 = exact1("SV")
        res1 = LeverageSHAP(n, random_state=seed).approximate(2**n, game1)
        np.testing.assert_allclose(res1.values, exact_sv1.values, atol=1e-8, rtol=0.0)

        # Additive game
        weights = np.arange(1.0, n + 1.0)

        def additive_game(Z, weights=weights):
            return Z.astype(float) @ weights

        exact2 = ExactComputer(additive_game, n)
        exact_sv2 = exact2("SV")
        res2 = LeverageSHAP(n, random_state=seed).approximate(2**n, additive_game)
        np.testing.assert_allclose(res2.values, exact_sv2.values, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_null_player_axiom(seed):
    """Players who never affect the game should get zero Shapley value."""
    n = 6
    null_idx = 5

    def game(Z):
        # Depend only on players 0..4, ignore player 5
        return Z[:, :null_idx].astype(float).sum(axis=1)

    res = LeverageSHAP(n, random_state=seed).approximate(2**n, game)
    # value slot 0 is baseline, player entries start at index 1; check the null player
    np.testing.assert_allclose(res.values[1 + null_idx], 0.0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_minimal_budget_sweep(seed):
    """Verify LeverageSHAP runs and behaves sensibly for tiny budgets.

    This checks budgets at and near the minimal valid values for a small n.
    """
    n = 4
    budgets = [2, 3, 4, 5, 8]
    for b in budgets:
        res = LeverageSHAP(n, random_state=seed).approximate(b, DummyGame(n, interaction=(0, 1)))
        assert res.estimation_budget is not None
        # With the deterministic default, the realized evaluation count is exact (see
        # test_estimation_budget_matches_exact_formula) and capped at full enumeration.
        assert res.estimation_budget <= 2**n
        if b < 2**n:
            assert res.estimated is True
        else:
            assert res.estimated is False


def test_inf_game_values_raise():
    """A game returning Inf values must raise ValueError, not silently return NaN Shapley values.

    Before the fix, v0=inf and v_grand=inf caused efficiency_shift=nan (inf-inf),
    which propagated through the solver into the returned InteractionValues without
    any indication of failure.
    """
    n = 5

    def inf_game(Z):
        return np.full(len(Z), np.inf)

    approximator = LeverageSHAP(n, random_state=0)
    with pytest.raises(ValueError, match="finite game values"):
        approximator.approximate(budget=20, game=inf_game)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_constant_game_zero_svs(seed):
    """A constant game v(S) = c for all S must assign zero Shapley value to every player.

    This puts b = 0 in the regression system (all game values equal the baseline after
    centering), so the solver receives a zero target vector. The efficiency axiom must
    still hold (sum of SVs == v(N) - v({}) == 0).
    """
    n = 6
    c = 7.5  # arbitrary non-zero constant

    def constant_game(Z):
        return np.full(len(Z), c)

    result = LeverageSHAP(n, random_state=seed).approximate(budget=2**n, game=constant_game)

    np.testing.assert_allclose(result.values[1:], 0.0, atol=1e-10)
    assert result.values[1:].sum() == pytest.approx(0.0, abs=1e-10)


def test_sample_without_replacement_huge_pool_fallback(monkeypatch):
    """Use the randrange/set fallback when total exceeds sys.maxsize."""
    py_rng = random.Random(123)

    def _should_not_be_called(*args, **kwargs):
        msg = "random.sample path should not be used in huge-pool fallback"
        raise AssertionError(msg)

    # Make any accidental call to random.sample fail, so this test proves the fallback branch.
    monkeypatch.setattr(random.Random, "sample", _should_not_be_called)
    monkeypatch.setattr(leverageshap_module.sys, "maxsize", 5)

    sampled = LeverageSHAP._sample_without_replacement(total=10, k=3, py_rng=py_rng)

    assert len(sampled) == 3
    assert len(set(sampled)) == 3
    assert all(0 <= idx < 10 for idx in sampled)


def test_combo_empty_combination_returns_all_false():
    """_combo should return an all-false vector when s == 0."""
    z = LeverageSHAP._combo(n=7, s=0, i=0)

    assert z.dtype == bool
    assert z.shape == (7,)
    assert z.sum() == 0
    assert np.array_equal(z, np.zeros(7, dtype=bool))


@pytest.mark.parametrize(("n", "s"), [(5, 2), (6, 3), (7, 1), (7, 4), (4, 4)])
def test_combo_matches_itertools_lexicographic_order(n, s):
    """_combo (Algorithm 3) must reproduce itertools.combinations in lexicographic order.

    This exercises the load-bearing while-loop recursion (not just the s == 0 early
    return): for every index i in [0, C(n, s)) the returned boolean vector must mark
    exactly the players of the i-th lexicographic size-s combination.
    """
    total = math.comb(n, s)
    for i, expected_players in enumerate(itertools.combinations(range(n), s)):
        z = LeverageSHAP._combo(n=n, s=s, i=i)
        expected = np.zeros(n, dtype=bool)
        expected[list(expected_players)] = True
        assert z.dtype == bool
        assert z.sum() == s
        np.testing.assert_array_equal(z, expected)
    # sanity check: we enumerated exactly C(n, s) combinations
    assert i + 1 == total


@pytest.mark.parametrize(
    ("n", "m"),
    [(4, 8), (5, 12), (6, 20), (7, 100), (10, 50), (12, 40)],
)
def test_find_c_solves_equation_12(n, m):
    """_find_c must solve Eq. 12: m - 2 == sum_{s=1}^{n-1} min(C(n, s), 2c).

    The oversampling constant c drives the whole sampling rate; a regression here would
    silently shift the budget match without failing the efficiency/ordering tests.
    """
    c = LeverageSHAP._find_c(n, m)
    total = sum(min(math.comb(n, s), 2.0 * c) for s in range(1, n))
    assert total == pytest.approx(m - 2, abs=1e-6)


def test_find_c_boundary_cases():
    """_find_c returns 0.0 for the degenerate regimes with nothing to subsample."""
    # n < 2: no interior coalition sizes exist.
    assert LeverageSHAP._find_c(n=1, m=2) == 0.0
    # target <= 0: budget only covers the empty and grand coalitions.
    assert LeverageSHAP._find_c(n=6, m=2) == 0.0
    assert LeverageSHAP._find_c(n=6, m=1) == 0.0


def test_find_c_large_n_overflow_safe():
    """_find_c must not overflow for large n where C(n, n//2) exceeds float range."""
    n, m = 2000, 5000
    c = LeverageSHAP._find_c(n, m)
    # For a modest budget the small sizes dominate; c stays finite and positive.
    assert math.isfinite(c)
    assert c > 0.0
    total = sum(min(math.comb(n, s), 2.0 * c) for s in range(1, n))
    assert total == pytest.approx(m - 2, abs=1e-3)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_underdetermined_efficiency_axiom(seed):
    """When budget << n, the design matrix A has fewer rows than columns (underdetermined).

    lstsq returns the minimum-norm solution in this regime. Efficiency must still hold
    exactly because it is enforced algebraically via the efficiency_shift construction,
    independently of the regression solve.
    """
    n = 10
    # budget=4 gives only ~2 interior rows for n=10, far fewer than n columns
    budget = 4

    game = DummyGame(n, interaction=(0, 1))
    v_grand = game(np.ones((1, n), dtype=bool))[0]
    v_empty = game(np.zeros((1, n), dtype=bool))[0]

    result = LeverageSHAP(n, random_state=seed).approximate(budget=budget, game=game)

    assert result.values[1:].sum() == pytest.approx(v_grand - v_empty, abs=1e-8)


@pytest.mark.parametrize("seed", DIVERSE_SEEDS)
def test_negative_large_magnitude_game(seed):
    """Game with large-magnitude negative values should not degrade numerical precision.

    Tests that neither the IS weight computation nor the solver loses precision when
    game values span a large negative range, which exercises different floating-point
    paths than the positive skewed game.
    """
    n = 7
    scale = 1e5

    def large_negative_game(Z):
        # Additive game with large negative weights — exact SVs are known analytically.
        player_weights = -scale * np.arange(1, n + 1, dtype=float)
        return Z.astype(float) @ player_weights

    v_grand = large_negative_game(np.ones((1, n)))[0]
    v_empty = large_negative_game(np.zeros((1, n)))[0]

    result = LeverageSHAP(n, random_state=seed).approximate(budget=2**n, game=large_negative_game)

    # efficiency must hold to near machine precision (algebraic, not solver-dependent)
    assert result.values[1:].sum() == pytest.approx(v_grand - v_empty, rel=1e-10)

    # exact SVs for an additive game equal the player weights; verify ordering
    # (player n has the most negative SV, player 1 the least negative)
    svs = result.values[1:]
    for i in range(n - 1):
        assert svs[i] > svs[i + 1], (
            f"Expected sv[{i}] > sv[{i + 1}], got {svs[i]:.6f} vs {svs[i + 1]:.6f}"
        )
