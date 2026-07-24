"""Tests for permutation-walk Shapley-Taylor interaction approximation."""

from __future__ import annotations

from itertools import combinations
from math import comb

import jax.numpy as jnp
import pytest

from shapiq import (
    STII,
    SV,
    CallableGame,
    InsufficientSamplesError,
    PermutationSampling,
)

N_PLAYERS = 5
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


def cubic_from_masks(masks):
    return quadratic_from_masks(masks) + 1.5 * masks[..., 0] * masks[..., 1] * masks[..., 2]


def game_from(mask_fn):
    return CallableGame(
        fn=lambda c: mask_fn(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def seeds(order):
    return 2 + sum(comb(N_PLAYERS, size) for size in range(1, order))


def quantum(order):
    if order == 1:
        return N_PLAYERS - 1
    n_interactions = comb(N_PLAYERS, order)
    return (N_PLAYERS - order) + n_interactions * (2**order - 1)


def subset_mask(subset):
    mask = jnp.zeros((N_PLAYERS,), dtype=jnp.float32)
    if subset:
        mask = mask.at[jnp.asarray(subset)].set(1.0)
    return mask


def discrete_derivative(mask_fn, interaction, base):
    total = 0.0
    for length in range(len(interaction) + 1):
        for part in combinations(interaction, length):
            sign = (-1) ** (len(interaction) - length)
            total += sign * float(mask_fn(subset_mask((*base, *part))))
    return total


def brute_force_top_stii(mask_fn, interaction):
    order = len(interaction)
    others = [p for p in range(N_PLAYERS) if p not in interaction]
    total = 0.0
    for size in range(N_PLAYERS - order + 1):
        weight = (order / N_PLAYERS) / comb(N_PLAYERS - 1, size)
        for base in combinations(others, size):
            total += weight * discrete_derivative(mask_fn, interaction, base)
    return total


def test_lower_orders_are_exact_derivatives_at_empty():
    policy = PermutationSampling(game_from(cubic_from_masks), STII(order=3), random_state=0)
    estimate = policy.estimate(seeds(3) + quantum(3))
    for player in range(N_PLAYERS):
        assert jnp.allclose(estimate[(player,)], WEIGHTS[player], atol=1e-5)
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(estimate[(left, right)], PAIRS[left, right], atol=1e-5)


def test_top_order_pairs_exact_for_quadratic_after_one_walk():
    policy = PermutationSampling(
        game_from(quadratic_from_masks), STII(order=2), random_state=0
    )
    estimate = policy.estimate(seeds(2) + quantum(2))
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(estimate[(left, right)], PAIRS[left, right], atol=1e-5)


def test_top_order_triples_exact_for_cubic_after_one_walk():
    policy = PermutationSampling(game_from(cubic_from_masks), STII(order=3), random_state=1)
    estimate = policy.estimate(seeds(3) + quantum(3))
    for triple in combinations(range(N_PLAYERS), 3):
        expected = 1.5 if triple == (0, 1, 2) else 0.0
        assert jnp.allclose(estimate[triple], expected, atol=1e-5)


def test_top_order_converges_to_brute_force_stii():
    policy = PermutationSampling(game_from(cubic_from_masks), STII(order=2), random_state=2)
    estimate = policy.estimate(seeds(2) + 1500 * quantum(2))
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(
            estimate[pair], brute_force_top_stii(cubic_from_masks, pair), atol=0.1
        )


def test_order_one_stii_matches_sv_approximator_exactly():
    n_walks = 25
    stii = PermutationSampling(game_from(cubic_from_masks), STII(order=1), random_state=4)
    sv = PermutationSampling(game_from(cubic_from_masks), SV(), random_state=4)
    stii_estimate = stii.estimate(seeds(1) + n_walks * quantum(1))
    sv_estimate = sv.estimate(2 + n_walks * (N_PLAYERS - 1))
    for player in range(N_PLAYERS):
        assert jnp.allclose(stii_estimate[(player,)], sv_estimate[(player,)], atol=1e-6)


def test_efficiency_holds_exactly_for_quadratic_games():
    game = game_from(quadratic_from_masks)
    grand = quadratic_from_masks(jnp.ones((N_PLAYERS,), dtype=jnp.float32))
    empty = quadratic_from_masks(jnp.zeros((N_PLAYERS,), dtype=jnp.float32))
    explanation = PermutationSampling(game, STII(order=2), random_state=3).estimate(
        seeds(2) + quantum(2),
    )
    total = sum(
        float(explanation[interaction])
        for order in (1, 2)
        for interaction in combinations(range(N_PLAYERS), order)
    )
    assert jnp.allclose(total, grand - empty, atol=1e-4)


def test_empty_interaction_is_the_empty_coalition_value():
    game = game_from(quadratic_from_masks)
    empty = quadratic_from_masks(jnp.zeros((N_PLAYERS,), dtype=jnp.float32))
    explanation = PermutationSampling(game, STII(order=2), random_state=0).estimate(
        seeds(2) + quantum(2),
    )
    assert jnp.allclose(explanation[()], empty, atol=1e-6)


def test_explaining_before_first_completed_walk_raises():
    policy = PermutationSampling(
        game_from(quadratic_from_masks), STII(order=2), random_state=0
    )
    with pytest.raises(InsufficientSamplesError):
        policy.estimate(0)[(0, 1)]
    with pytest.raises(InsufficientSamplesError):
        policy.estimate(seeds(2) + quantum(2) - 1)[(0, 1)]


def test_seed_block_resumes_across_budget_splits():
    def make():
        return PermutationSampling(game_from(cubic_from_masks), STII(order=3), random_state=6)

    total = seeds(3) + quantum(3)
    whole = make().estimate(total)
    policy = make()
    split = policy.refine(policy.refine(policy.estimate(5), 9), total - 14)  # 14 < seeds(3) = 17
    assert split.evidence == whole.evidence
    for triple in combinations(range(N_PLAYERS), 3):
        assert jnp.allclose(split[triple], whole[triple], atol=1e-6)


def test_pending_samples_and_budget_splits():
    def make():
        return PermutationSampling(game_from(cubic_from_masks), STII(order=2), random_state=5)

    whole = make().estimate(seeds(2) + 4 * quantum(2))
    policy = make()
    split = policy.refine(policy.estimate(quantum(2) + 7), seeds(2) + 3 * quantum(2) - 7)
    assert split.evidence == whole.evidence
    with_bank = make().estimate(seeds(2) + 4 * quantum(2) + 9)
    assert with_bank.bank == 9
    assert with_bank.evidence.n_samples == whole.evidence.n_samples
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(with_bank[pair], whole[pair], atol=1e-6)
