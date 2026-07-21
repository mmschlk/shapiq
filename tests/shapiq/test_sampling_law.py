"""Tests for the LawfulSampler capability: samplers declare their marginal law."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from shapiq.coalitions import DenseCoalitionArray
from shapiq.sampling import (
    BanzhafKernelSampler,
    EmptyState,
    LawfulSampler,
    PairedSampler,
    PermutationSampler,
    ProductKernelSampler,
    ShapleyKernelSampler,
    SizeKernelSampler,
)

N_PLAYERS = 6


def powerset_coalitions(n_players: int) -> DenseCoalitionArray:
    codes = np.arange(2**n_players, dtype=np.uint32)
    masks = ((codes[:, None] >> np.arange(n_players)) & 1).astype(bool)
    return DenseCoalitionArray(jnp.asarray(masks))


def test_the_shapley_kernel_law_sums_to_one_over_its_support():
    sampler = ShapleyKernelSampler(N_PLAYERS)
    log_probs = sampler.log_probability(powerset_coalitions(N_PLAYERS))
    assert np.isclose(float(jnp.sum(jnp.exp(log_probs))), 1.0, atol=1e-6)
    # the empty and grand coalition are seeds, outside the sampled support
    assert float(log_probs[0]) == -np.inf
    assert float(log_probs[-1]) == -np.inf


def test_the_product_law_has_full_support():
    sampler = ProductKernelSampler(N_PLAYERS, 0.3)
    coalitions = powerset_coalitions(N_PLAYERS)
    log_probs = sampler.log_probability(coalitions)
    sizes = np.asarray(jnp.sum(jnp.asarray(coalitions.to_dense()), axis=-1))
    expected = sizes * np.log(0.3) + (N_PLAYERS - sizes) * np.log(0.7)
    assert np.allclose(np.asarray(log_probs), expected, atol=1e-6)
    assert np.isclose(float(jnp.sum(jnp.exp(log_probs))), 1.0, atol=1e-6)


def test_the_banzhaf_law_is_uniform():
    sampler = BanzhafKernelSampler(N_PLAYERS)
    log_probs = sampler.log_probability(powerset_coalitions(N_PLAYERS))
    assert np.allclose(np.asarray(log_probs), -N_PLAYERS * np.log(2.0), atol=1e-6)


def test_pairing_symmetrizes_an_asymmetric_law():
    weights = jnp.asarray([0.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.0])
    base = SizeKernelSampler(N_PLAYERS, weights)
    paired = PairedSampler(SizeKernelSampler(N_PLAYERS, weights))
    assert isinstance(paired, LawfulSampler)
    coalitions = powerset_coalitions(N_PLAYERS)
    complements = DenseCoalitionArray(~jnp.asarray(coalitions.to_dense()))
    expected = jnp.logaddexp(
        base.log_probability(coalitions),
        base.log_probability(complements),
    ) - jnp.log(2.0)
    assert np.allclose(np.asarray(paired.log_probability(coalitions)), np.asarray(expected))


def test_pairing_leaves_a_symmetric_law_unchanged():
    base = ShapleyKernelSampler(N_PLAYERS)
    paired = PairedSampler(ShapleyKernelSampler(N_PLAYERS))
    coalitions = powerset_coalitions(N_PLAYERS)
    assert np.allclose(
        np.asarray(paired.log_probability(coalitions)),
        np.asarray(base.log_probability(coalitions)),
        atol=1e-6,
    )


def test_walk_samplers_declare_no_law():
    assert not isinstance(PermutationSampler(N_PLAYERS), LawfulSampler)
    assert not isinstance(PairedSampler(PermutationSampler(N_PLAYERS)), LawfulSampler)
    assert isinstance(ShapleyKernelSampler(N_PLAYERS), LawfulSampler)
    assert isinstance(ProductKernelSampler(N_PLAYERS, 0.4), LawfulSampler)


def test_empirical_frequencies_follow_the_law():
    n_players = 4
    sampler = ShapleyKernelSampler(n_players, random_state=5)
    n_sampled = 4000
    coalitions, _ = sampler.sample(EmptyState(), sampler.n_seed_samples + n_sampled)
    dense = np.asarray(jnp.asarray(coalitions.to_dense()))[sampler.n_seed_samples :]
    codes = dense @ (1 << np.arange(n_players))
    frequencies = np.bincount(codes.astype(np.int64), minlength=2**n_players) / n_sampled
    law = np.exp(np.asarray(sampler.log_probability(powerset_coalitions(n_players))))
    assert np.allclose(frequencies, law, atol=0.02)
