"""Tests for permutation-walk Shapley value approximation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import (
    CallableGame,
    DenseCoalitionArray,
    InsufficientSamplesError,
    PermutationSamplingSV,
    UnsupportedGameError,
)

N_PLAYERS = 5
QUANTUM = N_PLAYERS - 1
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


def order_one_attributions(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


def test_additive_game_is_exact_after_one_walk():
    game = CallableGame(
        fn=lambda c: jnp.asarray(c.to_dense(), dtype=jnp.float32) @ WEIGHTS,
        n_players=N_PLAYERS,
    )
    explanation = PermutationSamplingSV.create(game, key=0).sample(QUANTUM).explain()
    assert jnp.allclose(order_one_attributions(explanation), WEIGHTS, atol=1e-6)


@pytest.mark.parametrize("budget", [QUANTUM, QUANTUM + 3, 7 * QUANTUM + 1, 100])
def test_efficiency_holds_for_any_budget(budget):
    game = quadratic_game()
    grand = game(DenseCoalitionArray(jnp.ones((N_PLAYERS,), dtype=bool)))
    empty = game(DenseCoalitionArray(jnp.zeros((N_PLAYERS,), dtype=bool)))
    explanation = PermutationSamplingSV.create(game, key=3).sample(budget).explain()
    assert jnp.allclose(jnp.sum(order_one_attributions(explanation)), grand - empty, atol=1e-4)


def test_estimate_converges_to_exact_shapley_values():
    approximator = PermutationSamplingSV.create(quadratic_game(), key=7)
    explanation = approximator.sample(3000 * QUANTUM).explain()
    assert jnp.allclose(order_one_attributions(explanation), EXACT_SV, atol=0.05)


def test_sampling_is_invariant_to_budget_splits():
    def make():
        return PermutationSamplingSV.create(quadratic_game(), key=11)

    split = make().sample(7).sample(2).sample(3 * QUANTUM)
    whole = make().sample(7 + 2 + 3 * QUANTUM)
    assert split.state == whole.state
    assert split.sampler.n_pending == whole.sampler.n_pending
    assert jnp.allclose(
        order_one_attributions(split.explain()),
        order_one_attributions(whole.explain()),
        atol=1e-6,
    )


def test_pending_samples_are_masked_until_their_walk_completes():
    def make():
        return PermutationSamplingSV.create(quadratic_game(), key=5)

    complete = make().sample(2 * QUANTUM)
    with_pending = make().sample(2 * QUANTUM + 3)
    assert with_pending.sampler.n_pending == 3
    assert with_pending.state.n_samples == complete.state.n_samples + 3
    assert jnp.allclose(
        order_one_attributions(with_pending.explain()),
        order_one_attributions(complete.explain()),
        atol=1e-6,
    )


def test_explaining_before_first_completed_walk_raises():
    approximator = PermutationSamplingSV.create(quadratic_game(), key=0)
    with pytest.raises(InsufficientSamplesError):
        approximator.explain()
    with pytest.raises(InsufficientSamplesError):
        approximator.sample(QUANTUM - 1).explain()


def test_sampling_quantum_and_pending_are_observable():
    approximator = PermutationSamplingSV.create(quadratic_game(), key=0)
    assert approximator.sampler.sampling_quantum == QUANTUM
    assert approximator.sampler.n_pending == 0
    assert approximator.sample(QUANTUM + 1).sampler.n_pending == 1
    assert approximator.sample(QUANTUM + 1).sample(QUANTUM - 1).sampler.n_pending == 0


def test_empty_interaction_returns_empty_coalition_value():
    game = quadratic_game()
    empty = game(DenseCoalitionArray(jnp.zeros((N_PLAYERS,), dtype=bool)))
    explanation = PermutationSamplingSV.create(game, key=0).sample(QUANTUM).explain()
    assert jnp.allclose(explanation(()), empty, atol=1e-6)


def test_history_rollback_restores_earlier_estimates():
    approximator = PermutationSamplingSV.create(quadratic_game(), key=13, track_history=True)
    early = approximator.sample(2 * QUANTUM)
    late = early.sample(3 * QUANTUM)
    assert jnp.allclose(
        order_one_attributions(late.rollback(1).explain()),
        order_one_attributions(early.explain()),
        atol=1e-6,
    )


def test_batched_targets_with_independent_walks():
    per_target_weights = jnp.stack([WEIGHTS, 2.0 * WEIGHTS, WEIGHTS - 1.0])

    def additive_batched(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return jnp.sum(per_target_weights[:, None, :] * masks, axis=-1)

    game = CallableGame(fn=additive_batched, n_players=N_PLAYERS, target_shape=(3,))
    explanation = PermutationSamplingSV.create(game, key=2).sample(QUANTUM).explain()
    assert explanation.shape == (3,)
    attributions = jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)
    assert jnp.allclose(attributions, per_target_weights, atol=1e-6)


def test_shared_samples_across_targets():
    per_target_weights = jnp.stack([WEIGHTS, 2.0 * WEIGHTS, WEIGHTS - 1.0])

    def additive_batched(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return jnp.sum(per_target_weights[:, None, :] * masks, axis=-1)

    game = CallableGame(fn=additive_batched, n_players=N_PLAYERS, target_shape=(3,))
    approximator = PermutationSamplingSV.create(game, key=2, sample_sharing=True)
    assert approximator.sampler.shared_target_shape == (1,)
    explanation = approximator.sample(QUANTUM).explain()
    assert explanation.shape == (3,)
    attributions = jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)
    assert jnp.allclose(attributions, per_target_weights, atol=1e-6)


def test_vector_valued_games_are_rejected():
    def vector_values(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return jnp.stack([masks @ WEIGHTS, masks @ WEIGHTS], axis=-1)

    game = CallableGame(fn=vector_values, n_players=N_PLAYERS)
    with pytest.raises(UnsupportedGameError):
        PermutationSamplingSV.create(game, key=0)
