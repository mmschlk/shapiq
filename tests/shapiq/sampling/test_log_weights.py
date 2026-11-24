"""The latest sampler weighs every coalition drawn so far, across steps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq.sampling._base import (
    BorderSizeWithReplacement,
    PairedBorderSizeWithReplacement,
    PairedSizeWithoutReplacement,
    PairedSizeWithReplacement,
    SizeWithoutReplacement,
    SizeWithReplacement,
    UniformWithoutReplacement,
)

N_PLAYERS = 6
# Sizes 1 and 5 hold 6 coalitions each and fill up first; sizes 0 and 6 are never drawn.
Q = jnp.array([0.0, 1 / 5, 1 / 8, 1 / 9, 1 / 8, 1 / 5, 0.0])
SAMPLERS = {
    "size": lambda keys: SizeWithReplacement(keys=keys, weights=Q),
    "paired": lambda keys: PairedSizeWithReplacement(keys=keys, weights=Q),
    "border": lambda keys: BorderSizeWithReplacement(keys=keys, weights=Q),
    "paired border": lambda keys: PairedBorderSizeWithReplacement(keys=keys, weights=Q),
    "without": lambda keys: SizeWithoutReplacement(keys=keys, weights=Q),
    "paired without": lambda keys: PairedSizeWithoutReplacement(keys=keys, weights=Q),
    "uniform without": lambda keys: UniformWithoutReplacement(keys=keys, n_players=N_PLAYERS),
}


def _numbers(coalitions: jax.Array) -> np.ndarray:
    rows = np.asarray(coalitions)
    return (rows * (1 << np.arange(rows.shape[-1]))).sum(axis=-1)


@pytest.mark.parametrize("make", SAMPLERS.values(), ids=SAMPLERS.keys())
def test_latest_log_weights_of_all_draws_sum_to_an_unbiased_estimate(make):
    """Do the latest weights of every draw so far give an unbiased sum, for every sampler?"""
    n_runs = 2000
    sampler = make(jax.random.split(jax.random.key(0), n_runs))
    samples = []
    for budget in (4, 10, 6, 20):
        sample, sampler = sampler.sample(budget)
        # A step's own weights are the latest sampler's weights of its coalitions.
        np.testing.assert_allclose(sample.log_weights, sampler.log_weights(sample.coalitions))
        samples.append(sample)
    coalitions = jnp.concatenate([sample.coalitions for sample in samples], axis=-2)
    weights = np.exp(np.asarray(sampler.log_weights(coalitions)))  # (n_runs, 40)
    values = np.random.default_rng(0).normal(size=2**N_PLAYERS)
    values[[0, -1]] = 0.0  # never drawn by the size samplers
    estimates = np.sum(weights * values[_numbers(coalitions)], axis=-1)
    standard_error = np.std(estimates) / np.sqrt(n_runs)

    assert abs(estimates.mean() - values.sum()) < 4 * standard_error


def test_border_with_replacement_enumerates_each_filled_size_once_across_small_steps():
    """Do small steps enumerate a filled size once its cumulative share fits, then skip it?"""
    sampler = BorderSizeWithReplacement(keys=jax.random.split(jax.random.key(1), 3), weights=Q)
    numbers, sizes, flags = [], [], []
    for _ in range(6):  # 8 units per step: room for one 6-coalition size at a time
        sample, sampler = sampler.sample(8)
        numbers.append(_numbers(sample.coalitions))  # (n_runs, 8)
        sizes.append(np.asarray(sample.coalitions.sum(axis=-1)))
        flags.append(np.asarray(sampler.enumerated))  # (n_runs, n_sizes)
    # The step after which each border size counts as enumerated; the same for every key.
    steps = {s: int(np.argmax([f[0, s] for f in flags])) for s in (1, N_PLAYERS - 1)}

    assert flags[-1].tolist() == 3 * [[False, True, False, False, False, True, False]]
    assert 0 < steps[1] < steps[N_PLAYERS - 1]  # a few steps first, then one size at a time
    for size, step in steps.items():
        every = sorted(x for x in range(2**N_PLAYERS) if x.bit_count() == size)
        for run in range(3):
            assert sorted(numbers[step][run][sizes[step][run] == size]) == every  # once each
            assert not any(np.any(later[run] == size) for later in sizes[step + 1 :])
