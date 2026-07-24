"""Tests for user-facing validation, observability, and error messages."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import SV, CallableGame, InsufficientSamplesError, PermutationSampling

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
    from_int = PermutationSampling(additive_game(), SV(), random_state=7)
    from_key = PermutationSampling(additive_game(), SV(), random_state=jax.random.key(7))
    budget = SEEDS + QUANTUM
    assert from_int.estimate(budget).evidence == from_key.estimate(budget).evidence


@pytest.mark.parametrize("bad", [None, 1.5, True, np.random.default_rng(0), "7"])
def test_random_state_rejects_other_types_at_construction(bad):
    with pytest.raises(TypeError, match="random_state must be an integer seed or a JAX PRNG key"):
        PermutationSampling(additive_game(), SV(), random_state=bad)


def test_share_samples_accepts_false_and_rejects_none():
    PermutationSampling(additive_game(), SV(), share_samples=False)
    with pytest.raises(TypeError, match="share_samples must be a bool"):
        PermutationSampling(additive_game(), SV(), share_samples=None)


def test_min_budget_matches_first_explainable_budget():
    policy = PermutationSampling(additive_game(), SV())
    assert policy.min_budget == SEEDS + QUANTUM
    short = policy.estimate(policy.min_budget - 1)
    with pytest.raises(InsufficientSamplesError):
        short[(0,)]
    policy.estimate(policy.min_budget)[(0,)]


def test_error_messages_teach_the_working_idiom():
    policy = PermutationSampling(additive_game(), SV())
    # a sub-seed budget banks without evidence; the planes teach the fix
    banked = policy.estimate(1)
    assert banked.bank == 1
    with pytest.raises(InsufficientSamplesError, match=r"estimate = policy\.estimate"):
        banked[(0,)]
    estimate = policy.refine(banked, SEEDS + QUANTUM - 1)
    with pytest.raises(IndexError, match="past the initial state"):
        estimate.evidence.rollback(len(estimate.evidence.history()))
    with pytest.raises(TypeError, match=r"estimate\[\(0,\)\]"):
        estimate[0]


def test_budget_type_errors_name_the_offending_type():
    policy = PermutationSampling(additive_game(), SV())
    with pytest.raises(TypeError, match="budget must be an integer, got float"):
        policy.estimate(10.0)


def test_fresh_estimates_are_observable():
    policy = PermutationSampling(additive_game(), SV())
    estimate = policy.estimate(0)
    assert estimate.evidence.n_samples == 0
    assert estimate.bank == 0
    assert estimate.spent == 0
    text = repr(estimate)
    assert "Estimate" in text
    assert "n_samples=0" in text
    assert "bank=0" in text
