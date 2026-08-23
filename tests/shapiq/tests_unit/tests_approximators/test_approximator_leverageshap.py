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
    # default deterministic_counts=True this is exact (see
    # test_per_size_counts_structural_properties for the exact-formula check); it must
    # equal the game's access count and never exceed full enumeration (2**n).
    assert sv_estimates.estimation_budget == game.access_counter
    assert sv_estimates.estimation_budget <= 2**n
    assert sv_estimates.estimated != (budget >= 2**n)

    # The access counter should be at most 2**n, since LeverageSHAP caps the budget at 2**n and does not make redundant calls.
    assert game.access_counter <= 2**n

    # Players 1 and 2 should be the most important (DummyGame interaction on
    # (1, 2)). DummyGame(n, interaction) is v(S) = |S|/n + [1, 2 in S]: an
    # additive "coalition size" game (Shapley value 1/n for every player) plus
    # a pure pairwise unanimity game on {1, 2} (closed-form Shapley value 1/2
    # for players 1 and 2, 0 for everyone else -- Shapley's own formula for a
    # unanimity game u_T, SV_i(u_T) = 1/|T| for i in T). Exact target for
    # players 1 and 2: 1/n + 1/2. A loose tolerance here (e.g. abs=0.15) would pass
    # even under a wrong leverage-weight formula, since this game happens to be
    # recovered essentially exactly by LeverageSHAP at both of this test's budgets
    # regardless of the WLS weights, as long as the sampled design has full column
    # rank -- verified empirically over all DIVERSE_SEEDS x {n=7, budget in (380,
    # 100)}, max observed error ~1e-15. See test_sample_weights_match_leverage_score_
    # formula for a test that isolates the weights directly instead of relying on
    # this game's forgiving structure.
    exact_target = 1.0 / n + 0.5
    assert sv_estimates[(1,)] == pytest.approx(exact_target, abs=1e-6)
    assert sv_estimates[(2,)] == pytest.approx(exact_target, abs=1e-6)

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


def test_per_size_counts_match_hand_computed_tables():
    """Per-size sampled row counts must match literal, hand-computed expected tables.

    These four (n, budget) cases were solved by hand from the class docstring's
    definition (Eq. 12 for c, then the largest-remainder rounding of each half-size's
    Binomial(C(n, s), min(1, 2c / C(n, s))) mean into a pair count m_s, doubled for the
    middle size since a "pair" there is two distinct same-size coalitions) -- not by
    re-running or reimplementing shapiq's own algorithm. The numbers are transcribed
    once as a literal table below so a shared bug between this test and the
    implementation cannot make both sides agree. Derivations:

    n=4, budget=10: target = budget - 2 = 8. C(4,1)=4, C(4,2)=6, C(4,3)=4. Solving
    sum(min(C(4,s), 2c)) = 8 with 2c <= 4 gives 2c = 8/3 (all three terms equal 2c and
    sum to 3*(8/3) = 8), i.e. c = 4/3. Half-sizes are s=1 (not middle) and s=2 (middle,
    n even). s=1: mu = 2c = 8/3 ~= 2.667, floor 2, remainder 0.667. s=2 (middle):
    pool = C(3,1) = 3, mu = 2c*s/n = (8/3)*(2/4) = 4/3 ~= 1.333, floor 1, remainder
    0.333. Floor sum = 3, target_pairs = (10-2)//2 = 4, shortfall = 1, which goes to
    s=1 (larger remainder) -> pairs {1: 3, 2: 1}. Row counts: size 1 and its complement
    size 3 each get 3 rows (one row per pair, non-middle); size 2 is self-complementary
    so its 1 pair contributes 2 rows. Expected row counts: {1: 3, 2: 2, 3: 3}.

    n=5, budget=9: target = 7. C(5,1)=5, C(5,2)=10, C(5,3)=10, C(5,4)=5. Solving with
    2c <= 5 gives 2c = 7/4 = 1.75 (four equal terms summing to 7), c = 0.875. n is odd,
    so there is no middle size and mu = 2c for every half-size (s=1, s=2 alike): both
    get floor(1.75) = 1 with the *same* remainder 0.75 -- an exact tie. Floor sum = 2,
    target_pairs = (9-2)//2 = 3, shortfall = 1, which must go to exactly one of the two
    tied sizes; the implementation's stable, ascending-s tie-break gives it to s=1 ->
    pairs {1: 2, 2: 1}. Row counts: {1: 2, 2: 1, 3: 1, 4: 2} (symmetric, since neither
    size is middle). This case also demonstrates the odd-budget floor (budget=9 is odd;
    the realized total is 2 + 2*3 = 8 rows -- this class always floors an odd budget
    down to the nearest even total rather than rounding up, as documented in
    ``_bernoulli_sample_deterministic``'s docstring).

    n=7, budget=15: target = 13. C(7,1)=7, C(7,2)=21, C(7,3)=35. Solving with 2c <= 7
    gives 2c = 13/6 ~= 2.1667 (six equal terms across half-sizes 1..3 and their
    complements 4..6, summing to 6*(13/6) = 13), c ~= 1.0833. n is odd, so mu = 2c for
    every half-size: s=1, s=2, s=3 each get floor(2.1667) = 2 with the *same* remainder
    0.1667. Floor sum = 2+2+2 = 6, target_pairs = (15-2)//2 = 6, shortfall = 0 -- no
    tie-break needed at all, so this case is unambiguous regardless of tie-break rule.
    Row counts: {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2} (12 interior rows + 2 for
    empty/grand = 14; budget=15 is odd, floored down by one row as in the n=5 case).

    n=6, budget=40: target = 38. C(6,1)=6, C(6,2)=15, C(6,3)=20 (middle, pool
    C(5,2)=10). Solving with 6 <= 2c <= 15 (so s=1 and its complement s=5 are already
    exhaustive at this 2c, while s=2/s=4 and the middle s=3 are not) gives
    12 + 3*(2c) = 38 -> 2c = 26/3 ~= 8.667, c ~= 4.333. s=1: C(6,1)=6 <= 8.667, so s=1
    (and s=5) are taken exhaustively: pairs = pool = 6. s=2: not exhaustive
    (15 > 8.667), mu = 2c ~= 8.667, floor 8, remainder 0.667. s=3 (middle): not
    exhaustive (20 > 8.667), pool = C(5,2) = 10, mu = 2c*3/6 = 4.333, floor 4, remainder
    0.333. Floor sum over the two non-exhaustive sizes = 8 + 4 = 12, plus the
    exhaustive size's fixed 6 = 18 total pairs before remainder-filling.
    target_pairs = (40-2)//2 = 19, shortfall = 1, which goes to s=2 (largest remainder
    among the non-exhaustive sizes; s=1 is skipped because it is already exhaustive)
    -> pairs {1: 6, 2: 9, 3: 4}. Row counts: size 1 and its complement size 5 each get
    6 rows; size 2 and its complement size 4 each get 9 rows; size 3 is
    self-complementary so its 4 pairs contribute 8 rows. Expected row counts:
    {1: 6, 2: 9, 3: 8, 4: 9, 5: 6}. This is the case where a small size (s=1, and by
    symmetry s=5) saturates its full pool while budget is still well short of
    2**n = 64.
    """
    cases: list[tuple[int, int, dict[int, int]]] = [
        (4, 10, {1: 3, 2: 2, 3: 3}),
        (5, 9, {1: 2, 2: 1, 3: 1, 4: 2}),
        (7, 15, {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}),
        (6, 40, {1: 6, 2: 9, 3: 8, 4: 9, 5: 6}),
    ]
    for n, budget, expected_counts in cases:
        approximator = LeverageSHAP(n, random_state=0)
        Z, _ = approximator._sample(budget)
        sizes = Z.sum(axis=1)
        counts = Counter(int(s) for s in sizes[2:])  # rows 0, 1 are empty/grand
        for s in range(1, n):
            assert counts.get(s, 0) == expected_counts.get(s, 0), (n, budget, s)


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10, 11, 12])
@pytest.mark.parametrize("budget_fraction", [0.1, 0.3, 0.6])
def test_per_size_counts_structural_properties(n, budget_fraction):
    """Per-size sampled counts must satisfy structural invariants that hold by
    construction, independent of the exact c-solving / largest-remainder formula:
    the realized row total matches the simple budget-arithmetic
    formula (a distinct, much simpler piece of logic than the per-size distribution
    algorithm), each size's count never exceeds its full combinatorial pool,
    complementary sizes are equally represented, the self-complementary middle size
    (even n) always has an even count under pairing_trick=True, no two sampled rows
    are identical, and the empty/grand coalitions are always present as rows 0/1.
    """
    budget = max(2, int(budget_fraction * 2**n))
    approximator = LeverageSHAP(n, random_state=0)
    Z, _ = approximator._sample(budget)

    # empty/grand rows are always present, in that order
    assert not Z[0].any()
    assert Z[1].all()

    sizes = Z.sum(axis=1)
    counts = Counter(int(s) for s in sizes[2:])

    # total rows match the simple budget-arithmetic formula (distinct from, and much
    # simpler than, the per-size distribution algorithm this test file's hand-computed
    # tables cover)
    expected_total = 2 + 2 * ((min(budget, 2**n) - 2) // 2)
    assert Z.shape[0] == expected_total
    assert sum(counts.values()) + 2 == expected_total

    # no duplicate rows
    assert len({tuple(row) for row in Z}) == Z.shape[0]

    for s in range(1, n):
        # pool bound: can never sample more than C(n, s) distinct size-s coalitions
        assert counts.get(s, 0) <= math.comb(n, s)
        # symmetry: a size and its complement are always equally represented
        assert counts.get(s, 0) == counts.get(n - s, 0)

    # the self-complementary middle size (even n only) is always paired with itself,
    # so pairing_trick=True (the default here) always yields an even count for it
    if n % 2 == 0:
        assert counts.get(n // 2, 0) % 2 == 0


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
def test_per_size_counts_exhaustive_at_full_budget(n):
    """At budget == 2**n every coalition of every size must appear exactly once.

    This needs no c-solving or largest-remainder logic to state or check -- it is a
    direct mathematical consequence of "budget covers every coalition", independent of
    the per-size distribution formula this test file's hand-computed tables cover.
    """
    budget = 2**n
    approximator = LeverageSHAP(n, random_state=0)
    Z, _ = approximator._sample(budget)
    sizes = Z.sum(axis=1)
    counts = Counter(int(s) for s in sizes[2:])
    for s in range(1, n):
        assert counts.get(s, 0) == math.comb(n, s)


def test_sample_weights_match_leverage_score_formula():
    r"""Direct, hand-derived check of the per-row IS weights ``_sample`` returns,
    on partial (non-exhaustive) budgets.

    The class docstring's Algorithm 1 step 3 gives the formula: each row's weight
    is ``w(||z||) / min(1, 2c * l_z)``, where ``l_z = 1/C(n, ||z||)`` is the
    leverage score and ``w(s) = (s-1)!(n-s-1)!/n!`` is the Shapley kernel weight.
    When size ``s``'s whole layer is exhaustive (``C(n, s) <= 2c``), ``min(1, ...)
    == 1`` and the weight is exactly the raw kernel weight ``w(s)``. Otherwise
    ``min(1, ...) == 2c/C(n, s)`` and the weight collapses algebraically to
    ``w(s) * C(n, s) / (2c) == 1/(s * (n-s) * 2c)`` (the binomial cancels). Every
    number below is computed from this formula and Eq. 12 by hand -- not by
    calling ``_find_c`` or reading ``_sample``'s implementation -- so this test
    guards against two distinct classes of bug that a purely structural test
    would miss: always taking the exhaustive-branch weight (silently dropping
    the IS correction and reducing to something KernelSHAP-like), and an extra
    normalization factor sneaking into the leverage-weight branch. No "exact at
    full budget" test in this file exercises the non-exhaustive branch at all,
    since every size is trivially exhaustive there.

    n=5, budget=12: target = budget - 2 = 10. C(5,1)=C(5,4)=5, C(5,2)=C(5,3)=10.
    Guessing 2c <= 5 makes every one of the four binomials non-exhaustive, so
    each contributes exactly 2c: 4*(2c) = 10 -> c = 5/4, 2c = 5/2 (consistent
    with the guess: 2.5 <= 5). No size is exhaustive, so every row uses the
    leverage-corrected weight 1/(s*(n-s)*2c): size 1 and its complement, size 4
    -> 1/(1*4*2.5) = 0.1; size 2 and its complement, size 3 -> 1/(2*3*2.5) =
    1/15.

    n=6, budget=40: target = 38. C(6,1)=C(6,5)=6, C(6,2)=C(6,4)=15, C(6,3)=20
    (middle). Guessing 6 <= 2c <= 15 makes sizes 1/5 exhaustive and 2/3/4 not:
    2*6 + 3*(2c) = 38 -> 2c = 26/3 (c = 13/3; consistent: 6 <= 26/3 <= 15).
    Sizes 1 and 5 (C(n,s)=6 <= 26/3) get the raw kernel weight
    w(1) = w(5) = 0!*4!/6! = 1/30. Sizes 2 and 4 (not exhaustive,
    s*(n-s) = 2*4 = 8) get 1/(8*(26/3)) = 3/208. Size 3, the middle
    (self-complementary, s*(n-s) = 3*3 = 9), gets 1/(9*(26/3)) = 1/78.
    """
    cases: list[tuple[int, int, dict[int, float]]] = [
        (5, 12, {1: 0.1, 2: 1.0 / 15, 3: 1.0 / 15, 4: 0.1}),
        (6, 40, {1: 1.0 / 30, 2: 3.0 / 208, 3: 1.0 / 78, 4: 3.0 / 208, 5: 1.0 / 30}),
    ]
    for n, budget, expected_weight_by_size in cases:
        approximator = LeverageSHAP(n, random_state=0)
        Z, weights = approximator._sample(budget)
        sizes = Z.sum(axis=1)

        # Empty/grand rows carry weight 0 -- they enter via the efficiency shift,
        # not the weighted regression.
        assert weights[0] == 0.0
        assert weights[1] == 0.0

        for s, w in zip(sizes[2:], weights[2:], strict=True):
            expected = expected_weight_by_size[int(s)]
            assert w == pytest.approx(expected, rel=1e-9, abs=1e-12), (n, budget, int(s), w)


def test_saturation_all_rows_distinct_and_exact_on_soum():
    """At budget >= 2**n, all 2**n distinct coalitions are sampled and the estimate is
    exact (not merely close) on a SOUM game.
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
    report ``estimated``, draw distinct rows, and be seed-reproducible.
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


def test_deterministic_counts_false_budget_varies_across_seeds():
    """Distinguishes a genuine Binomial ``deterministic_counts=False`` path from a
    bug that silently always runs the deterministic path regardless of the flag.

    Every check in ``test_deterministic_counts_false_smoke`` above (``estimated``,
    distinct rows, same-seed reproducibility, budget <= 2**n) holds equally
    whether or not the flag actually does anything, because the (buggy)
    always-deterministic path is itself a valid, reproducible, distinct-rows
    sampler. What actually distinguishes the two is the flag's defining
    property: with ``deterministic_counts=False``, each size's pair count is a
    genuine Binomial *draw* rather than its rounded expectation, so the
    *realized total row count itself* is stochastic and varies across seeds --
    unlike ``deterministic_counts=True``, whose realized budget is always the
    exact closed form ``2 + 2 * ((min(budget, 2**n) - 2) // 2)`` regardless of
    seed.
    """
    n, budget = 6, 20

    budgets_false = {
        LeverageSHAP(n, deterministic_counts=False, random_state=seed)
        .approximate(budget, DummyGame(n, interaction=(0, 1)))
        .estimation_budget
        for seed in DIVERSE_SEEDS
    }
    assert len(budgets_false) > 1, "Binomial-path realized budget must vary across seeds"

    budgets_true = {
        LeverageSHAP(n, deterministic_counts=True, random_state=seed)
        .approximate(budget, DummyGame(n, interaction=(0, 1)))
        .estimation_budget
        for seed in DIVERSE_SEEDS
    }
    expected_deterministic_budget = 2 + 2 * ((min(budget, 2**n) - 2) // 2)
    assert budgets_true == {expected_deterministic_budget}


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
    least one row's complement is absent (the "without paired sampling" ablation).
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


def test_unpaired_complement_draw_calls_independent_sample(monkeypatch):
    """With ``pairing_trick=False``, the complement-side draw (size ``n - s``) for
    each half-size ``s`` must be a fresh, independent call to
    ``_sample_without_replacement``, not a reuse of the primary (size ``s``)
    draw's index list.

    This is a statistical-independence contract from the class docstring
    ("drawn independently instead of in forced complementary pairs"), not a
    count/shape/existential property, so it needs a spy on the actual sampling
    calls rather than an assertion on ``_sample``'s output: reusing the same
    index list for both sides would decode into *different* concrete rows
    anyway (lexicographic combo indices for size ``s`` and size ``n - s`` are
    unrelated encodings), so it does not literally recreate forced complement
    pairs and would otherwise slip past every existing row-level check
    (per-size counts, no-duplicate-rows, and the existential "some complement
    is missing" check in the test above all still pass under that bug).

    Grouping calls by their ``total`` (pool size) argument rather than by call
    order matters: a bug where the complement side stops calling
    ``_sample_without_replacement`` at all (aliasing the primary draw's
    indices instead) makes each non-middle half-size collapse from 2 calls to
    1 -- a regression a purely pairwise same-vs-different check on adjacent
    list entries can miss (adjacent entries would then belong to two
    *different* half-sizes, which still differ "by chance" and would let the
    bug slip through). The call-count version below catches that regression
    where a pairwise-adjacent check would not.
    """
    # n=10, budget=80: every half-size's layer is non-exhaustive (verified by hand
    # from Eq. 12: C(10,1)=10 already exceeds 2c here), so every draw genuinely
    # goes through the random-index path rather than the "k >= total -> return
    # every index" shortcut (which would make two draws for the same size
    # trivially identical regardless of independence, not a useful check).
    n, budget = 10, 80

    calls: list[tuple[int, int, tuple[int, ...]]] = []
    orig = LeverageSHAP._sample_without_replacement

    def spy(total, k, py_rng):
        result = orig(total, k, py_rng)
        calls.append((total, k, tuple(result)))
        return result

    monkeypatch.setattr(LeverageSHAP, "_sample_without_replacement", staticmethod(spy))
    approximator = LeverageSHAP(n, pairing_trick=False, random_state=0)
    approximator._sample(budget)

    # C(10, s) for s in 1..5 (the half-sizes) are all distinct (10, 45, 120, 210,
    # 252), so grouping by `total` unambiguously identifies which half-size each
    # call belongs to. The middle size (s=5, n even) draws 2*count items in a
    # single call in the unpaired branch; every non-middle size draws `count`
    # items in exactly two independent calls (primary, then complement).
    by_total: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for total, k, result in calls:
        by_total.setdefault(total, []).append((k, result))

    assert len(by_total) >= 2, "Expected at least one non-middle half-size to be sampled"
    non_middle_groups = [group for group in by_total.values() if len(group) != 1]
    assert non_middle_groups, "Expected at least one non-middle half-size (2 calls)"
    found_independent_pair = False
    for group in non_middle_groups:
        assert len(group) == 2, (
            "Each non-middle half-size must make exactly two "
            "_sample_without_replacement calls (primary + complement)"
        )
        (k0, result0), (k1, result1) = group
        assert k0 == k1
        if result0 != result1:
            found_independent_pair = True
    assert found_independent_pair, (
        "Expected the complement draw to differ from the primary draw for at least one half-size"
    )


def test_unpaired_binomial_combination_budget_dup_reproducible_unbiased():
    """The fourth flag combination -- ``pairing_trick=False`` with
    ``deterministic_counts=False`` -- must work like the other three: both flags
    are independent booleans, so all four combinations need coverage, not just
    the two where they agree.

    Checks, all at once: the realized budget never exceeds the requested cap, sampled
    rows are unique (without-replacement sampling holds even in the unpaired+Binomial
    path), the same ``random_state`` reproduces identical output, and the estimator is
    unbiased-ish -- averaged over many seeds, the mean estimate is close to the exact
    Shapley values on a SOUM game within a tolerance appropriate for a Monte Carlo mean
    over ``len(DIVERSE_SEEDS)`` draws.
    """
    n, budget = 6, 40  # budget < 2**n == 64, so both counts and pairing are exercised

    # Budget respected and rows unique.
    Z, _ = LeverageSHAP(n, pairing_trick=False, deterministic_counts=False, random_state=0)._sample(
        budget
    )
    assert Z.shape[0] <= budget
    rows = [tuple(row) for row in Z.astype(bool)]
    assert len(rows) == len(set(rows)), "Unpaired Binomial-path rows must be distinct"

    # Reproducible with the same random_state.
    game_a = DummyGame(n, interaction=(1, 2))
    game_b = DummyGame(n, interaction=(1, 2))
    res_a = LeverageSHAP(
        n, pairing_trick=False, deterministic_counts=False, random_state=7
    ).approximate(budget, game_a)
    res_b = LeverageSHAP(
        n, pairing_trick=False, deterministic_counts=False, random_state=7
    ).approximate(budget, game_b)
    np.testing.assert_array_equal(res_a.values, res_b.values)
    assert res_a.estimation_budget == res_b.estimation_budget

    # Unbiased-ish: mean estimate over many seeds is close to the exact SVs.
    game = SOUM(n=n, n_basis_games=15, max_interaction_size=3, random_state=42)
    exact = ExactComputer(game, n)
    exact_sv = exact("SV").values[1:]

    estimates = []
    for seed in DIVERSE_SEEDS:
        approx = LeverageSHAP(n, pairing_trick=False, deterministic_counts=False, random_state=seed)
        res = approx.approximate(budget, game)
        estimates.append(res.values[1:])
    mean_estimate = np.mean(estimates, axis=0)
    np.testing.assert_allclose(mean_estimate, exact_sv, atol=0.75, rtol=0.0)


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
    """LeverageSHAP should have no larger mean error than KernelSHAP on average,
    and should in fact be close to exact here. A one-directional
    ``err_leverage <= err_kernel`` assertion alone, with no absolute bound,
    would pass even if LeverageSHAP's own weighting were broken, as long as it
    happened not to be worse than KernelSHAP on these particular seeds -- the
    absolute near-exactness bound below is a real strengthening of that check.

    We compare average L2 error (w.r.t. ExactComputer) across several seeds for
    both methods at the same budget. LeverageSHAP's structured sampling (extreme
    sizes taken exhaustively, pairing enforced) recovers this game -- additive
    plus one pairwise interaction, i.e. degree <= 2 -- essentially exactly at
    this budget regardless of *which* rows are drawn: empirically, mean error
    over all of ``DIVERSE_SEEDS`` is ~5e-16 (float precision), while KernelSHAP's
    is ~0.15 on the same seeds/budget/game. The absolute bound below (0.02) sits
    comfortably between the two.

    Note: on *this* game/budget the WLS problem happens to have an exact
    (zero-residual) solution at the true SV regardless of which valid positive
    weights are used, so a bug that always uses the raw Shapley-kernel weight
    instead of the IS-reweighted one would *not* move ``err_leverage`` off its
    ~1e-15 baseline, and this bound alone would not catch it. That class of
    weight-formula bug is instead caught directly and reliably by
    ``test_sample_weights_match_leverage_score_formula``, which asserts the
    per-row weight values themselves. This test's absolute bound is kept as a
    genuine near-exact-recovery correctness property in its own right (and
    does catch row-selection/pairing bugs that break the design's coverage),
    not as a substitute for the direct weight check.
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
    assert err_leverage < 0.02


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
        # test_per_size_counts_structural_properties) and capped at full enumeration.
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
    silently shift the budget match without failing the efficiency/ordering tests. The
    tolerance is tightened to 1e-10: `_find_c`'s own bisection converges to
    `|hi - lo| < 1e-12 * max(1, hi)`, and `lo`/the midpoint solve Eq. 12 to
    within that same ~1e-12 tolerance too (the equation is smooth near the root),
    so a looser `abs=1e-6` tolerance would be ~4 orders of magnitude too loose to
    prefer `hi` over a converged `lo`/midpoint -- see
    test_find_c_saturates_hard_inequality_at_full_budget below for the actual
    (exact, tolerance-free) property that distinguishes them.
    """
    c = LeverageSHAP._find_c(n, m)
    total = sum(min(math.comb(n, s), 2.0 * c) for s in range(1, n))
    assert total == pytest.approx(m - 2, abs=1e-10)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 10, 12])
def test_find_c_saturates_hard_inequality_at_full_budget(n):
    """At full budget, `_find_c` must return the bisection's *upper* bound `hi`,
    not the converged midpoint or lower bound `lo`.

    `_find_c`'s own docstring/comment explains `hi` is returned deliberately so
    that at ``m == 2**n`` every size's layer is exhaustively covered: the
    bisection invariant ``total(hi) >= target`` holds at every iteration, so
    ``2 * hi`` must be at least the largest binomial coefficient
    ``max_s C(n, s)`` (the peak, hardest-to-saturate layer) exactly, with no
    floating-point slack. `lo` and the midpoint carry no such guarantee -- they
    can land a floating-point hair below the peak, silently failing to saturate
    it -- and unlike the equation-residual check above, this exact inequality
    has no continuity loophole: it is checked directly against the (exact,
    integer) binomial coefficient rather than against Eq. 12's smooth objective,
    so no tolerance is needed at all.
    """
    c = LeverageSHAP._find_c(n, 2**n)
    max_binom = max(math.comb(n, s) for s in range(1, n))
    assert 2.0 * c >= max_binom


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


def test_bernoulli_sample_degenerate_and_exhaustive_branches():
    """_bernoulli_sample must handle its two edge regimes correctly.

    With deterministic_counts=False (the legacy Binomial-count path), c <= 0.0 or
    n < 2 must short-circuit to an empty design (no interior sizes to sample), and
    a large c relative to n must exhaustively enumerate a size once 2c meets or
    exceeds C(n, s), rather than drawing a Binomial count that could exceed it.
    """
    approx = LeverageSHAP(n=6, deterministic_counts=False, random_state=0)

    # c <= 0.0: nothing to sample.
    z, sizes = approx._bernoulli_sample(n=6, c=0.0)
    assert z.shape == (0, 6)
    assert sizes.shape == (0,)

    # n < 2: no interior coalition sizes exist at all.
    z, sizes = approx._bernoulli_sample(n=1, c=1.0)
    assert z.shape == (0, 1)
    assert sizes.shape == (0,)

    # c large enough that 2c >= C(6, 1): size 1 must be exhaustively enumerated.
    z, sizes = approx._bernoulli_sample(n=6, c=6.5)
    assert (sizes == 1).sum() == math.comb(6, 1)


def test_bernoulli_sample_poisson_fallback_for_huge_pool():
    """_bernoulli_sample must fall back to a Poisson draw when a size's pool of
    coalitions exceeds the int32 range that np.random.Generator.binomial supports.

    n=40 puts C(40, 20) well above 2**31 - 1, so the middle size must use the
    Poisson-mean fallback rather than Binomial(pool_size, p).
    """
    assert math.comb(40, 20) > 2**31 - 1
    approx = LeverageSHAP(n=40, deterministic_counts=False, random_state=0)
    z, sizes = approx._bernoulli_sample(n=40, c=5.0)
    assert z.shape[0] == sizes.shape[0]
    assert z.shape[1] == 40


def test_bernoulli_sample_deterministic_exhaustive_skip_in_remainder_fill():
    """_bernoulli_sample_deterministic's largest-remainder fill loop must skip a
    size once it is already exhaustive, even if a frac-tie would otherwise select
    it for one more unit of budget.

    n=4, c=2.0, m=16 lands exactly on full enumeration (all 2**4 = 16 coalitions
    including empty/full sets), which forces the remainder-fill loop to visit an
    already-exhaustive size and take its `continue` branch instead of
    over-allocating past that size's pool.
    """
    approx = LeverageSHAP(n=4, random_state=0)
    z, sizes = approx._bernoulli_sample_deterministic(n=4, c=2.0, m=16)
    assert z.shape[0] == sizes.shape[0]
    for s in range(1, 4):
        assert (sizes == s).sum() == math.comb(4, s)


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
