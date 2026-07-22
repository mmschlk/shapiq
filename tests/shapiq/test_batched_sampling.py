"""Tests pinning batched draw generation bit-identical and stateless."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import PairedSampler
from shapiq.sampling import (
    BanzhafKernelSampler,
    PermutationSampler,
    ProductKernelSampler,
    ShapleyKernelSampler,
    SizeKernelSampler,
)

N_PLAYERS = 6


def all_samplers():
    return {
        "shapley-kernel": ShapleyKernelSampler(N_PLAYERS, random_state=3),
        "banzhaf-kernel": BanzhafKernelSampler(N_PLAYERS, random_state=3),
        "permutation": PermutationSampler(N_PLAYERS, random_state=3),
        "paired-shapley-kernel": PairedSampler(ShapleyKernelSampler(N_PLAYERS, random_state=3)),
        "paired-permutation": PairedSampler(PermutationSampler(N_PLAYERS, random_state=3)),
        "shapley-kernel-targets": ShapleyKernelSampler(N_PLAYERS, (2,), random_state=1),
        "permutation-targets": PermutationSampler(N_PLAYERS, (2,), random_state=1),
        "product-kernel": ProductKernelSampler(N_PLAYERS, 0.3, random_state=2),
        "size-kernel": SizeKernelSampler(N_PLAYERS, jnp.arange(N_PLAYERS + 1.0), random_state=2),
    }


def sampler_params():
    return pytest.mark.parametrize(
        "sampler",
        list(all_samplers().values()),
        ids=list(all_samplers().keys()),
    )


@sampler_params()
def test_batched_draws_match_single_unit_draws(sampler):
    batch = sampler.draws(jnp.arange(5))
    assert batch.shape[0] == 5 * sampler.draws_per_unit
    for index in range(5):
        single = sampler.draws(jnp.asarray([index]))
        start = index * sampler.draws_per_unit
        assert jnp.array_equal(batch[start : start + sampler.draws_per_unit], single)


@sampler_params()
def test_draws_are_order_free(sampler):
    forward = sampler.draws(jnp.arange(4))
    reverse = sampler.draws(jnp.asarray([3, 2, 1, 0]))
    per_unit = sampler.draws_per_unit
    for unit in range(4):
        mirrored = 3 - unit
        assert jnp.array_equal(
            forward[unit * per_unit : (unit + 1) * per_unit],
            reverse[mirrored * per_unit : (mirrored + 1) * per_unit],
        )


@sampler_params()
def test_samplers_are_stateless_values(sampler):
    first = sampler.draws(jnp.arange(3))
    second = sampler.draws(jnp.arange(3))
    assert jnp.array_equal(first, second)
