"""Tests for deduplicated permutation sampling."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    CallableGame,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    SamplingStallWarning,
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


def quadratic_from_masks(masks):
    return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)


def quadratic_game():
    return CallableGame(
        fn=lambda c: quadratic_from_masks(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def recording_game(rows):
    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        rows.append(np.asarray(coalitions.to_dense()).reshape(-1, N_PLAYERS))
        return quadratic_from_masks(masks)

    return CallableGame(fn=fn, n_players=N_PLAYERS)


def order_one_attributions(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


@pytest.mark.parametrize(
    ("approximator_cls", "kwargs"),
    [
        (PermutationSamplingSV, {}),
        (PermutationSamplingSII, {"order": 2}),
        (PermutationSamplingSTII, {"order": 2}),
    ],
)
def test_estimates_identical_to_plain_sampling(approximator_cls, kwargs):
    budget = 25
    deduplicated = approximator_cls(quadratic_game(), random_state=1, deduplicate=True, **kwargs).sample(
        budget,
    )
    raw_samples = deduplicated.state.n_samples
    assert raw_samples > budget  # duplicates were appended as free evidence
    plain = approximator_cls(quadratic_game(), random_state=1, **kwargs).sample(raw_samples)
    assert deduplicated.state == plain.state
    assert jnp.allclose(
        order_one_attributions(deduplicated.explain()),
        order_one_attributions(plain.explain()),
        atol=1e-6,
    )


def test_game_sees_each_coalition_exactly_once_and_budget_counts_novel():
    rows = []
    approximator = PermutationSamplingSV(recording_game(rows), random_state=0, deduplicate=True)
    approximator = approximator.sample(12).sample(8)
    evaluated = np.concatenate(rows, axis=0)
    assert evaluated.shape[0] == 12 + 8
    assert np.unique(evaluated, axis=0).shape[0] == evaluated.shape[0]


def test_sampling_is_invariant_to_budget_splits():
    def make():
        return PermutationSamplingSV(quadratic_game(), random_state=11, deduplicate=True)

    split = make().sample(9).sample(2).sample(8)
    whole = make().sample(19)
    assert split.state == whole.state
    assert split.sampler.n_pending_samples == whole.sampler.n_pending_samples


def test_stall_warns_and_leaves_budget_unspent():
    rows = []
    approximator = PermutationSamplingSV(recording_game(rows), random_state=2, deduplicate=True)
    with pytest.warns(SamplingStallWarning):
        approximator = approximator.sample(200)
    evaluated = np.concatenate(rows, axis=0)
    n_distinct = np.unique(evaluated, axis=0).shape[0]
    assert n_distinct < 200  # budget could not be spent
    assert n_distinct <= 2**N_PLAYERS
    assert approximator.state.n_samples > n_distinct  # free duplicate walks were kept
    explanation = approximator.explain()  # estimates still work after a stall
    grand = quadratic_from_masks(jnp.ones((N_PLAYERS,), dtype=jnp.float32))
    empty = quadratic_from_masks(jnp.zeros((N_PLAYERS,), dtype=jnp.float32))
    assert jnp.allclose(jnp.sum(order_one_attributions(explanation)), grand - empty, atol=1e-4)


def test_rollback_and_resample_are_consistent():
    first = PermutationSamplingSV(
        quadratic_game(),
        random_state=3,
        track_history=True,
        deduplicate=True,
    ).sample(15)
    second = first.sample(10)
    assert second.rollback(1).state == first.state
    assert second.rollback(1).sample(10).state == second.state


def test_deduplication_requires_shared_samples():
    def batched(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return jnp.stack([quadratic_from_masks(masks)] * 3)

    game = CallableGame(fn=batched, n_players=N_PLAYERS, target_shape=(3,))
    with pytest.raises(ValueError, match="share_samples=True"):
        PermutationSamplingSV(game, deduplicate=True)


def test_deduplication_works_with_shared_batched_targets():
    scales = jnp.asarray([1.0, -2.0, 0.5])

    def scaled(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return scales[:, None] * quadratic_from_masks(masks)

    game = CallableGame(fn=scaled, n_players=N_PLAYERS, target_shape=(3,))
    deduplicated = PermutationSamplingSV(
        game,
        random_state=4,
        share_samples=True,
        deduplicate=True,
    ).sample(20)
    plain = PermutationSamplingSV(game, random_state=4, share_samples=True).sample(
        deduplicated.state.n_samples,
    )
    assert deduplicated.state == plain.state
