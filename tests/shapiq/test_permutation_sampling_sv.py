"""Tests for permutation-walk Shapley value estimation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import (
    SV,
    CallableGame,
    DenseCoalitionArray,
    InsufficientSamplesError,
    PermutationSampling,
)

N_PLAYERS = 5
QUANTUM = N_PLAYERS - 1
SEEDS = 2
WEIGHTS = jnp.asarray([0.7, -1.3, 0.1, 2.0, -0.4])
PAIRS = jnp.asarray(
    [
        [0.0, 0.5, -1.0, 0.0, 0.3],
        [0.5, 0.0, 0.2, -0.7, 0.0],
        [-1.0, 0.2, 0.0, 0.4, 0.9],
        [0.0, -0.7, 0.4, 0.0, -0.2],
        [0.3, 0.0, 0.9, -0.2, 0.0],
    ],
)
# for v(S) = sum_{i in S} w_i + sum_{i<j in S} M_ij the Shapley values are exact:
EXACT_SV = WEIGHTS + 0.5 * jnp.sum(PAIRS, axis=1)


def quadratic_value(coalitions):
    masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
    return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)


def quadratic_game():
    return CallableGame(fn=quadratic_value, n_players=N_PLAYERS)


def order_one_attributions(estimate):
    return jnp.stack([estimate[(player,)] for player in range(N_PLAYERS)], axis=-1)


def test_additive_game_is_exact_after_one_walk():
    game = CallableGame(
        fn=lambda c: jnp.asarray(c.to_dense(), dtype=jnp.float32) @ WEIGHTS,
        n_players=N_PLAYERS,
    )
    estimate = PermutationSampling(game, SV(), random_state=0).estimate(SEEDS + QUANTUM)
    assert jnp.allclose(order_one_attributions(estimate), WEIGHTS, atol=1e-6)


@pytest.mark.parametrize(
    "budget", [SEEDS + QUANTUM, SEEDS + QUANTUM + 3, SEEDS + 7 * QUANTUM + 1, 100]
)
def test_efficiency_holds_for_any_budget(budget):
    game = quadratic_game()
    grand = game(DenseCoalitionArray(jnp.ones((N_PLAYERS,), dtype=bool)))
    empty = game(DenseCoalitionArray(jnp.zeros((N_PLAYERS,), dtype=bool)))
    estimate = PermutationSampling(game, SV(), random_state=3).estimate(budget)
    assert jnp.allclose(jnp.sum(order_one_attributions(estimate)), grand - empty, atol=1e-4)


def test_estimate_converges_to_exact_shapley_values():
    policy = PermutationSampling(quadratic_game(), SV(), random_state=7)
    estimate = policy.estimate(SEEDS + 3000 * QUANTUM)
    assert jnp.allclose(order_one_attributions(estimate), EXACT_SV, atol=0.05)


def test_sampling_is_invariant_to_budget_splits():
    def make():
        return PermutationSampling(quadratic_game(), SV(), random_state=11)

    policy = make()
    split = policy.refine(policy.refine(policy.estimate(7), 2), 3 * QUANTUM)
    whole = make().estimate(7 + 2 + 3 * QUANTUM)
    assert split.evidence == whole.evidence
    assert split.bank == whole.bank
    assert jnp.allclose(
        order_one_attributions(split),
        order_one_attributions(whole),
        atol=1e-6,
    )


def test_partial_walk_budgets_are_banked_not_evaluated():
    def make():
        return PermutationSampling(quadratic_game(), SV(), random_state=5)

    complete = make().estimate(SEEDS + 2 * QUANTUM)
    with_bank = make().estimate(SEEDS + 2 * QUANTUM + 3)
    assert with_bank.bank == 3
    assert with_bank.evidence.n_samples == complete.evidence.n_samples
    assert jnp.allclose(
        order_one_attributions(with_bank),
        order_one_attributions(complete),
        atol=1e-6,
    )


def test_reading_before_first_completed_walk_raises():
    policy = PermutationSampling(quadratic_game(), SV(), random_state=0)
    with pytest.raises(InsufficientSamplesError):
        policy.estimate(0)[(0,)]
    with pytest.raises(InsufficientSamplesError):
        policy.estimate(QUANTUM - 1)[(0,)]


def test_unit_rows_bank_and_spend_are_observable():
    policy = PermutationSampling(quadratic_game(), SV(), random_state=0)
    assert policy.unit_rows == QUANTUM
    assert policy.n_seed_samples == SEEDS
    banked = policy.estimate(SEEDS + QUANTUM + 1)
    assert banked.bank == 1
    assert banked.spent == SEEDS + QUANTUM
    completed = policy.refine(banked, QUANTUM - 1)
    assert completed.bank == 0
    assert completed.spent == SEEDS + 2 * QUANTUM


def test_baseline_carries_the_empty_coalition_value():
    game = quadratic_game()
    empty = game(DenseCoalitionArray(jnp.zeros((N_PLAYERS,), dtype=bool)))
    estimate = PermutationSampling(game, SV(), random_state=0).estimate(SEEDS + QUANTUM)
    assert jnp.allclose(estimate.view.baseline, empty, atol=1e-6)
    assert jnp.allclose(estimate.as_game()[()], empty, atol=1e-6)


def test_rollback_restores_earlier_estimates():
    policy = PermutationSampling(quadratic_game(), SV(), random_state=13)
    early = policy.estimate(SEEDS + 2 * QUANTUM)
    late = policy.refine(early, 3 * QUANTUM)
    rolled = policy.at_evidence(late.evidence.rollback(1))
    assert jnp.allclose(
        order_one_attributions(rolled),
        order_one_attributions(early),
        atol=1e-6,
    )


def test_history_begins_at_first_evidence_state():
    policy = PermutationSampling(quadratic_game(), SV(), random_state=13)
    fresh = policy.estimate(0)
    assert len(fresh.evidence.history()) == 1
    with pytest.raises(InsufficientSamplesError):
        fresh[(0,)]
    grown = policy.refine(policy.estimate(SEEDS + 2 * QUANTUM), QUANTUM)
    states = grown.evidence.history()
    assert [state.n_samples for state in states] == [
        SEEDS + 2 * QUANTUM,
        SEEDS + 3 * QUANTUM,
    ]
    with pytest.raises(IndexError, match="past the initial state"):
        grown.evidence.rollback(2)


def test_batched_targets_with_independent_walks():
    per_target_weights = jnp.stack([WEIGHTS, 2.0 * WEIGHTS, WEIGHTS - 1.0])

    def additive_batched(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return jnp.sum(per_target_weights[:, None, :] * masks, axis=-1)

    game = CallableGame(fn=additive_batched, n_players=N_PLAYERS, target_shape=(3,))
    estimate = PermutationSampling(game, SV(), random_state=2).estimate(SEEDS + QUANTUM)
    assert estimate.view.shape == (3,)
    attributions = jnp.stack([estimate[(player,)] for player in range(N_PLAYERS)], axis=-1)
    assert jnp.allclose(attributions, per_target_weights, atol=1e-6)


def test_shared_samples_across_targets():
    per_target_weights = jnp.stack([WEIGHTS, 2.0 * WEIGHTS, WEIGHTS - 1.0])

    def additive_batched(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return jnp.sum(per_target_weights[:, None, :] * masks, axis=-1)

    game = CallableGame(fn=additive_batched, n_players=N_PLAYERS, target_shape=(3,))
    policy = PermutationSampling(game, SV(), random_state=2, share_samples=True)
    assert policy.sampler.shared_target_shape == (1,)
    estimate = policy.estimate(SEEDS + QUANTUM)
    assert estimate.view.shape == (3,)
    attributions = jnp.stack([estimate[(player,)] for player in range(N_PLAYERS)], axis=-1)
    assert jnp.allclose(attributions, per_target_weights, atol=1e-6)


def test_undeclared_vector_values_are_rejected():
    def vector_values(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return jnp.stack([masks @ WEIGHTS, masks @ WEIGHTS], axis=-1)

    game = CallableGame(fn=vector_values, n_players=N_PLAYERS)  # value_shape not declared
    with pytest.raises(ValueError, match="declare value_shape"):
        PermutationSampling(game, SV(), random_state=0).estimate(16)
