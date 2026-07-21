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
    ExactExplainer,
    PairedSampler,
    PermutationSampling,
    ShapleyKernelSampler,
)
from shapiq.sampling import UnitScheduleSampler

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


def order_one(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


@pytest.mark.parametrize("sampler_type", [ShapleyKernelSampler, BanzhafKernelSampler])
def test_paired_kernel_units_are_the_wrapped_draw_plus_its_complement(sampler_type):
    paired = PairedSampler(sampler_type(N_PLAYERS, random_state=3))
    plain = sampler_type(N_PLAYERS, random_state=3)
    assert paired.sampling_quantum == 2 * plain.sampling_quantum
    for unit in range(5):
        unit_masks = paired._sampled_unit_masks(unit)
        base = plain._sampled_unit_masks(unit)
        assert unit_masks.shape[-2] == 2
        assert jnp.array_equal(unit_masks[..., 0, :], base[..., 0, :])
        assert jnp.array_equal(unit_masks[..., 1, :], ~base[..., 0, :])


def test_hookless_samplers_get_row_complement_pairing_for_free():
    class _FixedUnit(UnitScheduleSampler):
        @property
        def sampling_quantum(self):
            return 1

        def _sampled_unit_masks(self, unit_index):
            mask = jnp.arange(self.n_players) < (unit_index % self.n_players)
            return mask[None, :]

    paired = PairedSampler(_FixedUnit(N_PLAYERS))
    unit = paired._sampled_unit_masks(2)
    assert jnp.array_equal(unit[1], ~unit[0])


def test_pairing_twice_is_rejected_with_a_teaching_error():
    base = ShapleyKernelSampler(N_PLAYERS)
    with pytest.raises(TypeError, match="already paired"):
        PairedSampler(PairedSampler(base))


def test_paired_permutation_units_walk_the_reversed_permutation():
    approximator = PermutationSampling(quadratic_game(), SV(), paired=True, random_state=2)
    walk = N_PLAYERS - 1
    assert approximator.sampler.sampling_quantum == 2 * walk
    assert approximator.sampler.plan.length == walk
    evolved = approximator.sample(approximator.min_budget)
    masks = jnp.asarray(evolved.state.coalitions.to_dense())
    chain = masks[2 : 2 + walk]
    antithetic = masks[2 + walk : 2 + 2 * walk]
    # prefixes of the reversed permutation are reversed complements of the chain
    for row in range(walk):
        assert jnp.array_equal(antithetic[row], ~chain[walk - 1 - row])


def test_paired_shapley_value_walks_stay_exactly_efficient():
    approximator = PermutationSampling(quadratic_game(), SV(), paired=True, random_state=0)
    explanation = approximator.sample(approximator.min_budget).explain()
    grand = quadratic_from_masks(jnp.ones(N_PLAYERS, dtype=jnp.float32))
    empty = quadratic_from_masks(jnp.zeros(N_PLAYERS, dtype=jnp.float32))
    assert jnp.allclose(jnp.sum(order_one(explanation)), grand - empty, atol=1e-4)


def test_paired_streams_are_split_invariant():
    whole = PermutationSampling(quadratic_game(), SII(order=2), paired=True, random_state=5)
    split = PermutationSampling(quadratic_game(), SII(order=2), paired=True, random_state=5)
    budget = whole.min_budget + 7
    whole = whole.sample(budget)
    split = split.sample(3).sample(budget - 10).sample(7)
    assert whole.state == split.state


def test_paired_permutation_sampling_matches_the_exact_values():
    exact = order_one(ExactExplainer(quadratic_game(), SV()).explain())
    approximator = PermutationSampling(quadratic_game(), SV(), paired=True, random_state=7)
    estimate = order_one(approximator.sample(2 + 1500 * (N_PLAYERS - 1)).explain())
    assert jnp.allclose(estimate, exact, atol=0.05)


def test_paired_taylor_walks_explain():
    approximator = PermutationSampling(quadratic_game(), STII(order=2), paired=True, random_state=1)
    explanation = approximator.sample(approximator.min_budget).explain()
    assert explanation.order == 2
    assert explanation.interaction_index == "STII"
