"""Tests pinning batched unit generation bit-identical to the sequential stream."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import PairedSampler
from shapiq.sampling import (
    BanzhafKernelSampler,
    EmptyState,
    PermutationSIISampler,
    PermutationSTIISampler,
    ProductKernelSampler,
    ShapleyKernelSampler,
    SizeKernelSampler,
    UnitScheduleSampler,
)

N_PLAYERS = 6


class _FixedUnit(UnitScheduleSampler):
    """Custom sampler without a batched override; exercises the default path."""

    @property
    def sampling_quantum(self):
        return 3

    def _sampled_unit_masks(self, unit_index):
        mask = jnp.arange(self.n_players) < (unit_index % self.n_players)
        return jnp.stack([mask, ~mask, jnp.roll(mask, 1)])


def all_samplers():
    return {
        "shapley-kernel": ShapleyKernelSampler(N_PLAYERS, random_state=3),
        "banzhaf-kernel": BanzhafKernelSampler(N_PLAYERS, random_state=3),
        "permutation-sii": PermutationSIISampler(N_PLAYERS, order=2, random_state=3),
        "permutation-stii": PermutationSTIISampler(N_PLAYERS, order=2, random_state=3),
        "paired-shapley-kernel": PairedSampler(ShapleyKernelSampler(N_PLAYERS, random_state=3)),
        "paired-permutation-sii": PairedSampler(
            PermutationSIISampler(N_PLAYERS, order=2, random_state=3),
        ),
        "paired-permutation-stii": PairedSampler(
            PermutationSTIISampler(N_PLAYERS, order=2, random_state=3),
        ),
        "custom-default-batch": _FixedUnit(N_PLAYERS),
        "paired-custom": PairedSampler(_FixedUnit(N_PLAYERS)),
        "shapley-kernel-targets": ShapleyKernelSampler(N_PLAYERS, (2,), random_state=1),
        "permutation-sii-targets": PermutationSIISampler(
            N_PLAYERS, (2,), order=2, random_state=1
        ),
        "product-kernel": ProductKernelSampler(N_PLAYERS, 0.3, random_state=2),
        "size-kernel": SizeKernelSampler(
            N_PLAYERS, jnp.arange(N_PLAYERS + 1.0), random_state=2
        ),
    }


def sampler_params():
    return pytest.mark.parametrize(
        "sampler", list(all_samplers().values()), ids=list(all_samplers().keys())
    )


def reference_stream(sampler, budget):
    """Rebuild the schedule stream from scalar per-unit renders only."""
    chunks = []
    unit = 0
    total = 0
    while total < budget:
        masks = sampler._unit_masks(unit)
        chunks.append(masks)
        total += masks.shape[-2]
        unit += 1
    return jnp.concatenate(chunks, axis=-2)[..., :budget, :]


@sampler_params()
def test_batched_units_match_scalar_units(sampler):
    batch = sampler._sampled_unit_batch(jnp.arange(5))
    assert batch.shape[0] == 5
    for index in range(5):
        assert jnp.array_equal(batch[index], sampler._sampled_unit_masks(index))


@sampler_params()
def test_split_budgets_replay_the_scalar_reference_stream(sampler):
    total = sampler.n_seed_samples + 3 * sampler.sampling_quantum + 1
    expected = reference_stream(sampler, total)
    first, evolved = sampler.sample(EmptyState(), 3)
    second, evolved = evolved.sample(EmptyState(), total - 3)
    stream = jnp.concatenate(
        [jnp.asarray(first.to_dense()), jnp.asarray(second.to_dense())], axis=-2
    )
    assert jnp.array_equal(stream, expected)
    # the pending unit resumes bit-identically
    tail, _ = evolved.sample(EmptyState(), 2 * sampler.sampling_quantum)
    expected_tail = reference_stream(sampler, total + 2 * sampler.sampling_quantum)
    assert jnp.array_equal(jnp.asarray(tail.to_dense()), expected_tail[..., total:, :])


def test_batched_permutation_draws_match_scalar_draws():
    sampler = PermutationSIISampler(N_PLAYERS, order=2, random_state=7)
    draws = sampler.unit_draws(jnp.arange(4))
    for index in range(4):
        assert jnp.array_equal(draws[index], sampler.unit_draw(index))
