"""Tests for the ShaplEIG approximator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("botorch")

from shapiq.approximator.shapleig import ShaplEIG
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues

N_PLAYERS = 7
INTERACTIONS = {(), (1,), (1, 2), (0, 3, 4)}


def dummy_game(X):
    """Deterministic sum-of-unanimity game over N_PLAYERS players."""
    return np.array(
        [sum(1.0 for interaction in INTERACTIONS if all(x[i] for i in interaction)) for x in X]
    )


@pytest.fixture(scope="module")
def exact_sv():
    """Exact Shapley values of the dummy game."""
    exact_computer = ExactComputer(game=dummy_game, n_players=N_PLAYERS)
    return exact_computer(index="SV", order=1)


def test_initialization_defaults():
    """ShaplEIG initializes with the documented defaults."""
    approximator = ShaplEIG(n=9)
    assert approximator.n == 9
    assert approximator.index == "SV"
    assert approximator.max_order == 1
    assert approximator.initial_design_size == 10  # n + 1
    assert approximator.max_candidates == 1024
    assert approximator.valid_indices == ("SV",)


def test_approximate_output_format():
    """The output is a valid order-1 SV InteractionValues object."""
    approximator = ShaplEIG(n=N_PLAYERS, random_state=0, show_progress=False)
    estimates = approximator.approximate(budget=20, game=dummy_game)

    assert isinstance(estimates, InteractionValues)
    assert estimates.index == "SV"
    assert estimates.max_order == 1
    assert estimates.min_order == 0
    assert estimates.n_players == N_PLAYERS
    assert estimates.estimated
    assert estimates.estimation_budget == 20
    # the empty coalition carries the baseline value
    assert estimates[()] == pytest.approx(estimates.baseline_value)
    assert estimates.baseline_value == pytest.approx(1.0)  # () in INTERACTIONS


def test_determinism():
    """Same random_state -> identical estimates."""

    def run():
        return ShaplEIG(n=N_PLAYERS, random_state=42, show_progress=False).approximate(
            budget=20, game=dummy_game
        )

    first, second = run(), run()
    for i in range(N_PLAYERS):
        assert first[(i,)] == second[(i,)]


def test_convergence_to_exact(exact_sv):
    """Estimates approach the exact Shapley values as the budget grows."""
    errors = {}
    for budget in (15, 100):
        approximator = ShaplEIG(n=N_PLAYERS, random_state=0, show_progress=False)
        estimates = approximator.approximate(budget=budget, game=dummy_game)
        errors[budget] = float(
            np.mean([abs(estimates[(i,)] - exact_sv[(i,)]) for i in range(N_PLAYERS)])
        )
    # near-exhaustive budget (100 of 2^7 = 128 coalitions) must be accurate
    assert errors[100] < 1e-2
    assert errors[100] <= errors[15]


def test_budget_must_exceed_initial_design():
    """Budgets not exceeding the initial design are rejected."""
    approximator = ShaplEIG(n=N_PLAYERS, show_progress=False)
    with pytest.raises(ValueError, match="must exceed"):
        approximator.approximate(budget=N_PLAYERS + 1, game=dummy_game)


def test_exhaustive_candidates_require_small_n():
    """`max_candidates=None` (exhaustive candidates) is limited to n <= 16."""
    approximator = ShaplEIG(n=17, max_candidates=None, show_progress=False)
    with pytest.raises(ValueError, match="max_candidates"):
        approximator.approximate(budget=20, game=dummy_game)


def test_max_candidates_is_clamped_to_exhaustive():
    """`max_candidates` is an upper bound: small games fall back to exhaustive."""
    approximator = ShaplEIG(n=N_PLAYERS, max_candidates=1024, show_progress=False)
    # 2^7 - 8 = 120 candidates exist; a budget needing 121 iterations must
    # report the clamped (exhaustive) candidate count, not 1024.
    with pytest.raises(ValueError, match="only 120 candidate"):
        approximator.approximate(budget=129, game=dummy_game)


def test_sampled_candidate_subset():
    """A sampled candidate subset (`max_candidates`) yields valid estimates."""
    approximator = ShaplEIG(n=N_PLAYERS, max_candidates=32, random_state=0, show_progress=False)
    estimates = approximator.approximate(budget=15, game=dummy_game)
    assert isinstance(estimates, InteractionValues)
    assert estimates.estimated


def test_warmstart_run():
    """Warmstarted hyperparameter refits yield valid estimates."""
    approximator = ShaplEIG(n=N_PLAYERS, warmstart=True, random_state=0, show_progress=False)
    estimates = approximator.approximate(budget=15, game=dummy_game)
    assert isinstance(estimates, InteractionValues)
