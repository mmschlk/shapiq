"""Tests for deduplicated sampling through the frozen-policy verbs."""

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
    Regression,
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


def order_one_attributions(estimate):
    return jnp.stack([estimate[(player,)] for player in range(N_PLAYERS)], axis=-1)


@pytest.mark.parametrize("index", [SV(), SII(order=2), STII(order=2)])
def test_estimates_identical_to_plain_sampling(index):
    budget = 25
    deduplicated = PermutationSampling(
        quadratic_game(), index, random_state=1, deduplicate=True
    ).estimate(budget)
    raw_samples = deduplicated.evidence.n_samples
    assert raw_samples > budget  # duplicates were appended as free evidence
    plain = PermutationSampling(quadratic_game(), index, random_state=1).estimate(raw_samples)
    assert deduplicated.evidence == plain.evidence
    assert jnp.allclose(
        order_one_attributions(deduplicated),
        order_one_attributions(plain),
        atol=1e-6,
    )


def test_game_sees_each_coalition_exactly_once_and_budget_counts_novel():
    rows = []
    policy = PermutationSampling(recording_game(rows), SV(), random_state=0, deduplicate=True)
    estimate = policy.refine(policy.estimate(12), 8)
    evaluated = np.concatenate(rows, axis=0)
    # whole-unit spending: the final unit may overshoot into the bank, and
    # spent + bank always balances the budgets handed in
    assert evaluated.shape[0] == estimate.spent
    assert estimate.spent + estimate.bank == 12 + 8
    assert np.unique(evaluated, axis=0).shape[0] == evaluated.shape[0]


def test_sampling_is_invariant_to_budget_splits():
    def make():
        return PermutationSampling(quadratic_game(), SV(), random_state=11, deduplicate=True)

    policy = make()
    split = policy.refine(policy.refine(policy.estimate(9), 2), 8)
    whole = make().estimate(19)
    assert split.evidence == whole.evidence
    assert split.bank == whole.bank


def test_stall_warns_and_leaves_budget_unspent():
    rows = []
    policy = PermutationSampling(recording_game(rows), SV(), random_state=2, deduplicate=True)
    with pytest.warns(SamplingStallWarning):
        estimate = policy.estimate(200)
    evaluated = np.concatenate(rows, axis=0)
    n_distinct = np.unique(evaluated, axis=0).shape[0]
    assert n_distinct < 200  # budget could not be spent
    assert n_distinct <= 2**N_PLAYERS
    assert estimate.evidence.n_samples > n_distinct  # free duplicate walks were kept
    grand = quadratic_from_masks(jnp.ones((N_PLAYERS,), dtype=jnp.float32))
    empty = quadratic_from_masks(jnp.zeros((N_PLAYERS,), dtype=jnp.float32))
    # estimates still work after a stall
    assert jnp.allclose(jnp.sum(order_one_attributions(estimate)), grand - empty, atol=1e-4)


def test_branches_from_a_shared_parent_stay_consistent():
    policy = PermutationSampling(quadratic_game(), SV(), random_state=6, deduplicate=True)
    parent = policy.estimate(12)
    first = policy.refine(policy.refine(parent, 9), 7)
    second = policy.refine(policy.refine(parent, 9), 7)  # estimates are inert: branch freely
    fresh_policy = PermutationSampling(quadratic_game(), SV(), random_state=6, deduplicate=True)
    fresh = fresh_policy.refine(fresh_policy.refine(fresh_policy.estimate(12), 9), 7)
    assert first.evidence == fresh.evidence
    assert second.evidence == fresh.evidence


def test_rollback_and_resample_are_consistent():
    policy = PermutationSampling(quadratic_game(), SV(), random_state=3, deduplicate=True)
    first = policy.estimate(15)
    second = policy.refine(first, 10)
    rolled = policy.at_evidence(second.evidence.rollback(1))
    assert rolled.evidence == first.evidence
    assert policy.refine(rolled, 10).evidence == second.evidence


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
    ).estimate(20)
    plain = PermutationSampling(game, SV(), random_state=4, share_samples=True).estimate(
        deduplicated.evidence.n_samples,
    )
    assert deduplicated.evidence == plain.evidence


def test_banking_only_calls_checkpoint_and_replay_bit_identically():
    policy = PermutationSampling(quadratic_game(), SV(), random_state=3, deduplicate=True)
    first = policy.estimate(15)  # borrow leaves a negative bank
    second = policy.refine(first, 10)
    # a call whose whole budget repays the borrow still checkpoints
    repaid = policy.refine(first, 1)
    unchanged = policy.at_evidence(repaid.evidence.rollback(0))
    assert unchanged.evidence == repaid.evidence
    assert unchanged.bank == repaid.bank
    assert policy.at_evidence(repaid.evidence.rollback(1)).bank == first.bank
    # rollback restores the exact resume point: replaying the same budget
    # reproduces the same evidence and bank
    replayed = policy.refine(policy.at_evidence(second.evidence.rollback(1)), 10)
    assert replayed.evidence == second.evidence
    assert replayed.bank == second.bank
    assert replayed.spent == second.spent


def test_stalled_sampling_is_split_invariant_and_stops_growing():
    def tiny_game():
        return CallableGame(
            fn=lambda c: jnp.sum(jnp.asarray(c.to_dense(), dtype=jnp.float32), axis=-1),
            n_players=2,
        )

    def make():
        return Regression(tiny_game(), SV(), random_state=0, share_samples=True, deduplicate=True)

    with pytest.warns(SamplingStallWarning):
        whole = make().estimate(30)
    policy = make()
    with pytest.warns(SamplingStallWarning):
        split = policy.refine(policy.estimate(15), 15)
    assert whole.evidence == split.evidence
    assert whole.bank == split.bank
    # an exhausted estimate banks further budgets without growing
    with pytest.warns(SamplingStallWarning):
        again = make().refine(whole, 5)
    assert again.evidence.n_samples == whole.evidence.n_samples
    assert again.bank == whole.bank + 5


def test_rollback_replays_exactly_across_a_stall_boundary():
    # the stall counter is derived from the evidence, so a rolled-back
    # estimate resumes exactly like the original run did - even when the
    # roll crosses a quiet stall
    def small_game():
        return CallableGame(
            fn=lambda c: jnp.sum(jnp.asarray(c.to_dense(), dtype=jnp.float32), axis=-1) ** 2,
            n_players=4,
        )

    policy = Regression(small_game(), SV(), random_state=3, share_samples=True, deduplicate=True)
    with pytest.warns(SamplingStallWarning):
        stalled = policy.estimate(14)
    with pytest.warns(SamplingStallWarning):
        grown = policy.refine(stalled, 20)
    rolled = policy.at_evidence(grown.evidence.rollback(1))
    with pytest.warns(SamplingStallWarning):
        replayed = policy.refine(rolled, 20)
    assert replayed.evidence == grown.evidence
    assert replayed.bank == grown.bank
