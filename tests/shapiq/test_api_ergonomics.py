"""Tests for user-facing validation, observability, and error messages."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import CallableGame, HistoryError, InsufficientSamplesError, PermutationSamplingSV

N_PLAYERS = 5
QUANTUM = N_PLAYERS - 1
SEEDS = 2
WEIGHTS = jnp.asarray([0.7, -1.3, 0.1, 2.0, -0.4])


def additive_game():
    return CallableGame(
        fn=lambda c: jnp.asarray(c.to_dense(), dtype=jnp.float32) @ WEIGHTS,
        n_players=N_PLAYERS,
    )


def test_random_state_accepts_int_and_prng_key():
    from_int = PermutationSamplingSV(additive_game(), random_state=7)
    from_key = PermutationSamplingSV(additive_game(), random_state=jax.random.key(7))
    assert from_int.sample(SEEDS + QUANTUM).state == from_key.sample(SEEDS + QUANTUM).state


@pytest.mark.parametrize("bad", [None, 1.5, True, np.random.default_rng(0), "7"])
def test_random_state_rejects_other_types_at_construction(bad):
    with pytest.raises(TypeError, match="random_state must be an integer seed or a JAX PRNG key"):
        PermutationSamplingSV(additive_game(), random_state=bad)


def test_share_samples_accepts_false_and_rejects_none():
    PermutationSamplingSV(additive_game(), share_samples=False)
    with pytest.raises(TypeError, match="share_samples must be a bool"):
        PermutationSamplingSV(additive_game(), share_samples=None)


def test_track_history_must_be_a_bool():
    with pytest.raises(TypeError, match="track_history must be a bool"):
        PermutationSamplingSV(additive_game(), track_history=1)


def test_min_budget_matches_first_explainable_budget():
    approximator = PermutationSamplingSV(additive_game())
    assert approximator.min_budget == SEEDS + QUANTUM
    with pytest.raises(InsufficientSamplesError):
        approximator.sample(approximator.min_budget - 1).explain()
    approximator.sample(approximator.min_budget).explain()


def test_error_messages_teach_the_working_idiom():
    approximator = PermutationSamplingSV(additive_game())
    with pytest.raises(InsufficientSamplesError, match=r"approximator = approximator\.sample"):
        approximator.explain()
    with pytest.raises(InsufficientSamplesError, match=r"sample at least \d+ evaluations"):
        approximator.sample(3).explain()
    with pytest.raises(HistoryError, match="track_history=True"):
        approximator.history()
    explanation = approximator.sample(SEEDS + QUANTUM).explain()
    with pytest.raises(TypeError, match=r"explanation\(\(0,\)\)"):
        explanation(0)


def test_budget_type_errors_name_the_offending_type():
    approximator = PermutationSamplingSV(additive_game())
    with pytest.raises(TypeError, match="budget must be an integer, got float"):
        approximator.sample(10.0)


def test_fresh_approximator_is_observable():
    approximator = PermutationSamplingSV(additive_game())
    assert approximator.state.n_samples == 0
    text = repr(approximator)
    assert "PermutationSamplingSV" in text
    assert "n_samples=0" in text
    assert "n_pending_samples=0" in text
