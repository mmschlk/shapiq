"""Tests for permutation-walk Shapley interaction approximation."""

from __future__ import annotations

from itertools import combinations
from math import factorial

import jax.numpy as jnp
import pytest

from shapiq import (
    SII,
    SV,
    CallableGame,
    InsufficientSamplesError,
    PermutationSampling,
)

N_PLAYERS = 5
SII_QUANTUM = 2 * (N_PLAYERS - 1)
SEEDS = 2
N_PAIRS = N_PLAYERS * (N_PLAYERS - 1) // 2
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


def game_from(mask_fn, target_shape=()):
    return CallableGame(
        fn=lambda c: mask_fn(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
        target_shape=target_shape,
    )


def subset_mask(subset):
    mask = jnp.zeros((N_PLAYERS,), dtype=jnp.float32)
    if subset:
        mask = mask.at[jnp.asarray(subset)].set(1.0)
    return mask


def brute_force_pair_sii(mask_fn, pair):
    others = [p for p in range(N_PLAYERS) if p not in pair]
    total = 0.0
    for size in range(N_PLAYERS - 1):
        weight = factorial(size) * factorial(N_PLAYERS - 2 - size) / factorial(N_PLAYERS - 1)
        for subset in combinations(others, size):
            base = subset_mask(subset)
            both = subset_mask((*subset, *pair))
            first = subset_mask((*subset, pair[0]))
            second = subset_mask((*subset, pair[1]))
            derivative = mask_fn(both) - mask_fn(first) - mask_fn(second) + mask_fn(base)
            total += weight * float(derivative)
    return total


def pair_attributions(explanation):
    return {pair: explanation(pair) for pair in combinations(range(N_PLAYERS), 2)}


def test_pair_estimates_are_exact_for_quadratic_games():
    explanation = (
        PermutationSampling(game_from(quadratic_from_masks), SII(), random_state=0)
        .sample(SEEDS + 50 * SII_QUANTUM)
        .explain()
    )
    for (left, right), attribution in pair_attributions(explanation).items():
        assert jnp.allclose(attribution, PAIRS[left, right], atol=1e-4)


def test_pair_estimates_converge_to_brute_force_sii():
    explanation = (
        PermutationSampling(game_from(cubic_from_masks), SII(), random_state=1)
        .sample(SEEDS + 4000 * SII_QUANTUM)
        .explain()
    )
    for pair, attribution in pair_attributions(explanation).items():
        assert jnp.allclose(attribution, brute_force_pair_sii(cubic_from_masks, pair), atol=0.1)


def test_order_one_estimates_match_sv_approximator_exactly():
    n_walks = 30
    sii = PermutationSampling(game_from(cubic_from_masks), SII(), random_state=4)
    sv = PermutationSampling(game_from(cubic_from_masks), SV(), random_state=4)
    sii_explanation = sii.sample(SEEDS + n_walks * SII_QUANTUM).explain()
    sv_explanation = sv.sample(SEEDS + n_walks * (N_PLAYERS - 1)).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(sii_explanation((player,)), sv_explanation((player,)), atol=1e-6)


def test_order_one_sii_matches_sv_approximator_exactly():
    n_walks = 25
    sii = PermutationSampling(game_from(cubic_from_masks), SII(order=1), random_state=9)
    sv = PermutationSampling(game_from(cubic_from_masks), SV(), random_state=9)
    sii_explanation = sii.sample(SEEDS + n_walks * (N_PLAYERS - 1)).explain()
    sv_explanation = sv.sample(SEEDS + n_walks * (N_PLAYERS - 1)).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(sii_explanation((player,)), sv_explanation((player,)), atol=1e-6)


def test_order_three_triples_are_exact_for_cubic_games():
    quantum = (N_PLAYERS - 1) + (N_PLAYERS - 1) * 1 + (N_PLAYERS - 2) * 4
    approximator = PermutationSampling(game_from(cubic_from_masks), SII(order=3), random_state=3)
    explanation = approximator.sample(SEEDS + 100 * quantum).explain()
    assert explanation.order == 3
    for triple in combinations(range(N_PLAYERS), 3):
        expected = 1.5 if triple == (0, 1, 2) else 0.0
        assert jnp.allclose(explanation(triple), expected, atol=1e-4)


def test_order_three_coverage_error_after_one_walk():
    quantum = (N_PLAYERS - 1) + (N_PLAYERS - 1) * 1 + (N_PLAYERS - 2) * 4
    approximator = PermutationSampling(
        game_from(quadratic_from_masks), SII(order=3), random_state=0
    )
    with pytest.raises(InsufficientSamplesError):
        approximator.sample(SEEDS + quantum).explain()


def test_explaining_before_pair_coverage_raises():
    approximator = PermutationSampling(game_from(quadratic_from_masks), SII(), random_state=0)
    with pytest.raises(InsufficientSamplesError):
        approximator.sample(SEEDS + SII_QUANTUM - 1).explain()
    with pytest.raises(InsufficientSamplesError):
        approximator.sample(SEEDS + SII_QUANTUM).explain()


def test_pending_samples_are_masked_until_their_walk_completes():
    def make():
        return PermutationSampling(game_from(quadratic_from_masks), SII(), random_state=5)

    complete = make().sample(SEEDS + 40 * SII_QUANTUM)
    with_pending = make().sample(SEEDS + 40 * SII_QUANTUM + 5)
    assert with_pending.sampler.n_pending_samples == 5
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(with_pending.explain()(pair), complete.explain()(pair), atol=1e-6)


def test_sampling_is_invariant_to_budget_splits():
    def make():
        return PermutationSampling(game_from(quadratic_from_masks), SII(), random_state=11)

    split = make().sample(13).sample(3).sample(30 * SII_QUANTUM - 16)
    whole = make().sample(30 * SII_QUANTUM)
    assert split.state == whole.state
    assert split.sampler.n_pending_samples == whole.sampler.n_pending_samples


def test_explanation_metadata_and_normalization():
    explanation = (
        PermutationSampling(game_from(quadratic_from_masks), SII(), random_state=0)
        .sample(SEEDS + 50 * SII_QUANTUM)
        .explain()
    )
    assert explanation.order == 2
    assert explanation.interaction_index == "SII"
    assert bool(jnp.all(explanation.has((0, 1))))
    assert not bool(jnp.any(explanation.has(())))
    assert jnp.allclose(explanation((3, 1)), explanation((1, 3)), atol=1e-7)


def test_shared_samples_across_scaled_targets():
    scales = jnp.asarray([1.0, -2.0, 0.5])

    def scaled_quadratic(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return scales[:, None] * quadratic_from_masks(masks)

    game = CallableGame(fn=scaled_quadratic, n_players=N_PLAYERS, target_shape=(3,))
    approximator = PermutationSampling(game, SII(), random_state=2, share_samples=True)
    explanation = approximator.sample(SEEDS + 50 * SII_QUANTUM).explain()
    assert explanation.shape == (3,)
    for (left, right), attribution in pair_attributions(explanation).items():
        assert attribution.shape == (3,)
        assert jnp.allclose(attribution, scales * PAIRS[left, right], atol=1e-4)
