"""Tests for the kernel sampler family: size kernels, product measures, stream pins."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from shapiq.interactions._indices import _shapley_regression_kernel
from shapiq.sampling import (
    BanzhafKernelSampler,
    EmptyState,
    ProductKernelSampler,
    ShapleyKernelSampler,
    SizeKernelSampler,
)

N_PLAYERS = 7


def sampled_rows(sampler, budget):
    coalitions, _ = sampler.sample(EmptyState(), budget)
    return jnp.asarray(coalitions.to_dense())


def test_shapley_stream_is_pinned_to_the_original_arithmetic():
    # the pre-family sampler: size via choice over arange(1, n), members via permutation
    key = jax.random.key(9)
    sizes = jnp.arange(1, N_PLAYERS)
    weights = 1.0 / (sizes * (N_PLAYERS - sizes))
    probabilities = weights / jnp.sum(weights)
    sampler = ShapleyKernelSampler(N_PLAYERS, random_state=9)
    for unit in range(20):
        unit_key = jax.random.fold_in(key, unit)
        size_key, member_key = jax.random.split(unit_key)
        size = jax.random.choice(size_key, sizes, shape=(), p=probabilities)
        players = jnp.arange(N_PLAYERS)
        permutation = jax.random.permutation(member_key, players, axis=-1, independent=True)
        expected = (jnp.argsort(permutation) < size)[None, :]
        assert jnp.array_equal(sampler._sampled_unit_masks(unit), expected)


def test_banzhaf_stream_is_pinned_to_fair_coin_flips():
    key = jax.random.key(4)
    sampler = BanzhafKernelSampler(N_PLAYERS, random_state=4)
    product = ProductKernelSampler(N_PLAYERS, 0.5, random_state=4)
    for unit in range(20):
        expected = jax.random.bernoulli(jax.random.fold_in(key, unit), 0.5, (N_PLAYERS,))
        assert jnp.array_equal(sampler._sampled_unit_masks(unit), expected[None, :])
        assert jnp.array_equal(product._sampled_unit_masks(unit), expected[None, :])


def test_size_kernel_samples_only_positive_weight_sizes():
    weights = jnp.zeros(N_PLAYERS + 1).at[3].set(1.0)
    sampler = SizeKernelSampler(N_PLAYERS, weights, random_state=0)
    rows = sampled_rows(sampler, 40)
    assert jnp.array_equal(jnp.sum(rows[2:], axis=-1), jnp.full(38, 3))


def test_size_kernel_mixes_declared_sizes_and_members():
    weights = jnp.zeros(N_PLAYERS + 1).at[jnp.asarray([1, 6])].set(1.0)
    sampler = SizeKernelSampler(N_PLAYERS, weights, random_state=1)
    sizes = jnp.sum(sampled_rows(sampler, 402)[2:], axis=-1)
    assert set(jnp.unique(sizes).tolist()) == {1, 6}
    # both sizes appear with roughly equal frequency
    assert 120 < int(jnp.sum(sizes == 1)) < 280


def test_from_coalition_kernel_matches_the_shapley_size_marginal():
    sampler = SizeKernelSampler.from_coalition_kernel(
        N_PLAYERS,
        _shapley_regression_kernel(N_PLAYERS),
    )
    reference = ShapleyKernelSampler(N_PLAYERS)
    assert jnp.array_equal(sampler._sizes, reference._sizes)
    assert jnp.allclose(sampler._size_probabilities, reference._size_probabilities, atol=1e-7)


def test_product_kernel_memberships_follow_the_probability():
    sampler = ProductKernelSampler(N_PLAYERS, 0.2, random_state=3)
    rows = sampled_rows(sampler, 2002)[2:]
    frequencies = jnp.mean(rows.astype(jnp.float32), axis=0)
    assert jnp.all(jnp.abs(frequencies - 0.2) < 0.05)


def test_size_kernel_validation_teaches():
    with pytest.raises(ValueError, match="one weight per coalition size"):
        SizeKernelSampler(N_PLAYERS, jnp.ones(N_PLAYERS))
    with pytest.raises(ValueError, match="non-negative and finite"):
        SizeKernelSampler(N_PLAYERS, jnp.ones(N_PLAYERS + 1).at[2].set(-1.0))
    with pytest.raises(ValueError, match="non-negative and finite"):
        SizeKernelSampler(N_PLAYERS, jnp.ones(N_PLAYERS + 1).at[2].set(jnp.inf))
    with pytest.raises(ValueError, match="at least one coalition size"):
        SizeKernelSampler(N_PLAYERS, jnp.zeros(N_PLAYERS + 1))


def test_product_kernel_validation_teaches():
    for bad in (0.0, 1.0, -0.3, 1.5):
        with pytest.raises(ValueError, match="0 < p < 1"):
            ProductKernelSampler(N_PLAYERS, bad)
    with pytest.raises(TypeError, match="p must be a float"):
        ProductKernelSampler(N_PLAYERS, "half")
