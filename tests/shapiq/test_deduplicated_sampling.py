"""Tests for deduplicated permutation sampling."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    SII,
    STII,
    SV,
    CallableGame,
    PermutationSampling,
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


@pytest.mark.parametrize("index", [SV(), SII(order=2), STII(order=2)])
def test_estimates_identical_to_plain_sampling(index):
    budget = 25
    deduplicated = PermutationSampling(
        quadratic_game(), index, random_state=1, deduplicate=True
    ).sample(
        budget,
    )
    raw_samples = deduplicated.state.n_samples
    assert raw_samples > budget  # duplicates were appended as free evidence
    plain = PermutationSampling(quadratic_game(), index, random_state=1).sample(raw_samples)
    assert deduplicated.state == plain.state
    assert jnp.allclose(
        order_one_attributions(deduplicated.explain()),
        order_one_attributions(plain.explain()),
        atol=1e-6,
    )


def test_game_sees_each_coalition_exactly_once_and_budget_counts_novel():
    rows = []
    approximator = PermutationSampling(recording_game(rows), SV(), random_state=0, deduplicate=True)
    approximator = approximator.sample(12).sample(8)
    evaluated = np.concatenate(rows, axis=0)
    # whole-unit spending: the final unit may overshoot into the bank, and
    # spent + bank always balances the budgets handed in
    assert evaluated.shape[0] == approximator.spent
    assert approximator.spent + approximator.bank == 12 + 8
    assert np.unique(evaluated, axis=0).shape[0] == evaluated.shape[0]


def test_sampling_is_invariant_to_budget_splits():
    def make():
        return PermutationSampling(quadratic_game(), SV(), random_state=11, deduplicate=True)

    split = make().sample(9).sample(2).sample(8)
    whole = make().sample(19)
    assert split.state == whole.state
    assert split.bank == whole.bank


def test_stall_warns_and_leaves_budget_unspent():
    rows = []
    approximator = PermutationSampling(recording_game(rows), SV(), random_state=2, deduplicate=True)
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


def test_branches_from_a_shared_parent_stay_consistent():
    def make():
        return PermutationSampling(quadratic_game(), SV(), random_state=6, deduplicate=True)

    parent = make().sample(12)
    first = parent.sample(9).sample(7)
    second = parent.sample(9).sample(7)  # branches after `first` extended the carried keys
    fresh = make().sample(12).sample(9).sample(7)
    assert first.state == fresh.state
    assert second.state == fresh.state


def test_rollback_and_resample_are_consistent():
    first = PermutationSampling(
        quadratic_game(), SV(), random_state=3, deduplicate=True
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
        PermutationSampling(game, SV(), deduplicate=True)


def test_deduplication_works_with_shared_batched_targets():
    scales = jnp.asarray([1.0, -2.0, 0.5])

    def scaled(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return scales[:, None] * quadratic_from_masks(masks)

    game = CallableGame(fn=scaled, n_players=N_PLAYERS, target_shape=(3,))
    deduplicated = PermutationSampling(
        game, SV(), random_state=4, share_samples=True, deduplicate=True
    ).sample(20)
    plain = PermutationSampling(game, SV(), random_state=4, share_samples=True).sample(
        deduplicated.state.n_samples,
    )
    assert deduplicated.state == plain.state
