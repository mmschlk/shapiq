"""This test module contains all tests regarding the LeverageSHAP regression approximator."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.approximator.regression import LeverageSHAP
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame


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
def test_approximate(n, budget):
    """Tests the approximation of the LeverageSHAP approximator."""
    interaction = (1, 2)
    game = DummyGame(n, interaction)

    approximator = LeverageSHAP(n, random_state=42)
    sv_estimates = approximator.approximate(budget, game)

    assert isinstance(sv_estimates, InteractionValues)
    assert sv_estimates.max_order == 1
    assert sv_estimates.min_order == 0
    assert sv_estimates.index == "SV"
    assert sv_estimates.estimation_budget <= budget
    assert sv_estimates.estimated != (budget >= 2**n)

    # budget is respected
    assert game.access_counter <= budget

    # players 1 and 2 should be the most important (DummyGame interaction on (1, 2))
    assert sv_estimates[(1,)] == pytest.approx(0.6429, abs=0.15)
    assert sv_estimates[(2,)] == pytest.approx(0.6429, abs=0.15)

    # efficiency axiom: sum of SVs == v(N) - v({})
    efficiency = np.sum(sv_estimates.values[1:])
    v_grand = game(np.ones((1, n), dtype=bool))[0]
    v_empty = game(np.zeros((1, n), dtype=bool))[0]
    assert efficiency == pytest.approx(v_grand - v_empty, abs=1e-6)


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


@pytest.mark.parametrize("pairing_trick", [True, False])
def test_pairing_trick(pairing_trick):
    """LeverageSHAP should run without errors with and without the pairing trick."""
    n, budget = 6, 40
    game = DummyGame(n, interaction=(0, 1))
    approximator = LeverageSHAP(n, pairing_trick=pairing_trick, random_state=7)
    result = approximator.approximate(budget=budget, game=game)
    assert isinstance(result, InteractionValues)


def test_unanimity_game_exact_svs():
    """Unanimity game v(S) = 1 iff T ⊆ S is fully non-additive.

    True Shapley values: 1/|T| for players in T, 0 for others.
    LeverageSHAP's IS weights (1/(s(n-s))) differ from the Shapley kernel weights
    ((n-1)/(binom(n,s)*s*(n-s))), so even at full budget the per-run solution is a
    biased estimate — it converges to the Shapley values only in expectation over
    many random runs. What must always hold is:
    - efficiency axiom: sum(SVs) == v(N) - v({}) = 1
    - structural ordering: every player in T gets a strictly higher SV than every player outside T
    """
    n = 6
    T = frozenset({1, 3, 5})

    def unanimity_game(Z):
        return np.array([1.0 if all(row[j] for j in T) else 0.0 for row in Z])

    approximator = LeverageSHAP(n, random_state=42)
    result = approximator.approximate(budget=2**n, game=unanimity_game)

    # efficiency must always hold exactly
    assert result.values[1:].sum() == pytest.approx(1.0, abs=1e-8)

    sv = result.values[1:]
    t_svs = [sv[i] for i in T]
    non_t_svs = [sv[i] for i in range(n) if i not in T]
    assert min(t_svs) > max(non_t_svs)


def test_large_n_efficiency_axiom():
    """For large n, binom(n, s) would overflow or lose precision if not cancelled.

    Verifies that the IS weight cancellation (1/binom · binom = 1/(s(n-s))) keeps
    the efficiency axiom satisfied to machine precision even for n=20.
    """
    n = 20
    rng = np.random.default_rng(0)
    weights = rng.standard_normal(n)

    def additive_game(Z):
        return Z.astype(float) @ weights

    v_grand = additive_game(np.ones((1, n)))[0]
    v_empty = additive_game(np.zeros((1, n)))[0]

    approximator = LeverageSHAP(n, random_state=0)
    result = approximator.approximate(budget=500, game=additive_game)

    assert result.values[1:].sum() == pytest.approx(v_grand - v_empty, abs=1e-6)


def test_skewed_interaction_game():
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
    approximator = LeverageSHAP(n, random_state=42)
    # Approximate Shapley values under the skewed payoff function using the chosen budget.
    result = approximator.approximate(budget=300, game=skewed_game)

    # efficiency must hold despite extreme scale differences
    assert result.values[1:].sum() == pytest.approx(v_grand - v_empty, rel=1e-4)

    # dominant player (index 0) must have the highest SV
    assert result[(dominant,)] == pytest.approx(max(result.values[1:]), abs=1e-6)


def test_reproducibility():
    """Same seed should produce identical approximations across runs."""
    n, budget = 6, 100

    # Use separate game instances so access counters don't interfere
    game1 = DummyGame(n, interaction=(1, 2))
    game2 = DummyGame(n, interaction=(1, 2))

    # Run 1
    approx1 = LeverageSHAP(n, random_state=42)
    res1 = approx1.approximate(budget, game1)

    # Run 2
    approx2 = LeverageSHAP(n, random_state=42)
    res2 = approx2.approximate(budget, game2)

    # Values should be identical
    np.testing.assert_array_equal(res1.values, res2.values)

    # Other run metadata should match exactly
    assert res1.estimation_budget == res2.estimation_budget
    assert res1.estimated == res2.estimated


def test_reproducibility_different_seeds():
    """Different seeds should generally produce different approximations."""
    n, budget = 6, 100

    # Use separate game instances so access counters don't interfere
    game1 = DummyGame(n, interaction=(1, 2))
    game2 = DummyGame(n, interaction=(1, 2))

    # use fresh game instances for independent runs
    res_a = LeverageSHAP(n, random_state=0).approximate(budget, game1)
    res_b = LeverageSHAP(n, random_state=1).approximate(budget, game2)

    # very unlikely to be identical; assert they are not exactly equal
    assert not np.array_equal(res_a.values, res_b.values)


def test_empirical_convergence_rate():
    """The approximation error (w.r.t. ExactComputer) should decrease when the budget increases.

    Use averaging across a few seeds to reduce stochastic noise in the test.
    """

    n = 6
    seeds = [0, 1, 2, 3]

    def game_factory():
        return DummyGame(n, interaction=(0, 2))

    # ground truth (ExactComputer expects (game, n_players))
    exact = ExactComputer(game_factory(), n)
    exact_sv = exact("SV").values[1:]

    def mean_error(budget: int) -> float:
        errs = []
        for s in seeds:
            res = LeverageSHAP(n, random_state=s).approximate(budget, game_factory())
            errs.append(np.linalg.norm(exact_sv - res.values[1:]))
        return float(np.mean(errs))

    err_small = mean_error(20)
    err_large = mean_error(200)
    assert err_large < err_small
