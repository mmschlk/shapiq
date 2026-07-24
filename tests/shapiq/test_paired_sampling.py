"""Tests for draw-level pairing: complements for kernels, reversed walks for permutations."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import (
    SII,
    STII,
    SV,
    BanzhafKernelSampler,
    CallableGame,
    Estimate,
    ExactExplainer,
    PairedSampler,
    PermutationSampling,
    ShapleyKernelSampler,
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


def quadratic_game():
    return CallableGame(
        fn=lambda c: quadratic_from_masks(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def order_one(source):
    read = source.__getitem__ if isinstance(source, Estimate) else source
    return jnp.stack([read((player,)) for player in range(N_PLAYERS)], axis=-1)


@pytest.mark.parametrize("sampler_type", [ShapleyKernelSampler, BanzhafKernelSampler])
def test_paired_kernel_units_are_the_wrapped_draw_plus_its_complement(sampler_type):
    paired = PairedSampler(sampler_type(N_PLAYERS, random_state=3))
    plain = sampler_type(N_PLAYERS, random_state=3)
    assert paired.draws_per_unit == 2 * plain.draws_per_unit
    draws = paired.draws(jnp.arange(5))
    base = plain.draws(jnp.arange(5))
    for unit in range(5):
        assert jnp.array_equal(draws[2 * unit], base[unit])
        assert jnp.array_equal(draws[2 * unit + 1], ~base[unit])


def test_samplers_without_an_antithesis_are_rejected_with_a_teaching_error():
    class _NoAntithesis:
        n_players = N_PLAYERS

    with pytest.raises(TypeError, match="declares no antithesis"):
        PairedSampler(_NoAntithesis())


def test_pairing_twice_is_rejected_with_a_teaching_error():
    base = ShapleyKernelSampler(N_PLAYERS)
    with pytest.raises(TypeError, match="already paired"):
        PairedSampler(PairedSampler(base))


def test_paired_permutation_units_walk_the_reversed_permutation():
    policy = PermutationSampling(quadratic_game(), SV(), paired=True, random_state=2)
    walk = N_PLAYERS - 1
    assert policy.unit_rows == 2 * walk
    assert policy.sampler.draws_per_unit == 2
    evolved = policy.estimate(policy.min_budget)
    masks = jnp.asarray(evolved.evidence.coalitions.to_dense())
    chain = masks[2 : 2 + walk]
    antithetic = masks[2 + walk : 2 + 2 * walk]
    # prefixes of the reversed permutation are reversed complements of the chain
    for row in range(walk):
        assert jnp.array_equal(antithetic[row], ~chain[walk - 1 - row])


def test_paired_shapley_value_walks_stay_exactly_efficient():
    policy = PermutationSampling(quadratic_game(), SV(), paired=True, random_state=0)
    explanation = policy.estimate(policy.min_budget)
    grand = quadratic_from_masks(jnp.ones(N_PLAYERS, dtype=jnp.float32))
    empty = quadratic_from_masks(jnp.zeros(N_PLAYERS, dtype=jnp.float32))
    assert jnp.allclose(jnp.sum(order_one(explanation)), grand - empty, atol=1e-4)


def test_paired_streams_are_split_invariant():
    whole_policy = PermutationSampling(quadratic_game(), SII(order=2), paired=True, random_state=5)
    split_policy = PermutationSampling(quadratic_game(), SII(order=2), paired=True, random_state=5)
    budget = whole_policy.min_budget + 7
    whole = whole_policy.estimate(budget)
    split = split_policy.refine(split_policy.refine(split_policy.estimate(3), budget - 10), 7)
    assert whole.evidence == split.evidence


def test_paired_permutation_sampling_matches_the_exact_values():
    exact = order_one(ExactExplainer(quadratic_game(), SV()).explain())
    policy = PermutationSampling(quadratic_game(), SV(), paired=True, random_state=7)
    estimate = order_one(policy.estimate(2 + 1500 * (N_PLAYERS - 1)))
    assert jnp.allclose(estimate, exact, atol=0.05)


def test_paired_taylor_walks_explain():
    policy = PermutationSampling(quadratic_game(), STII(order=2), paired=True, random_state=1)
    explanation = policy.estimate(policy.min_budget)
    assert explanation.view.order == 2
    assert explanation.index == STII(order=2)
