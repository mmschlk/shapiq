"""Paired and border samplers stay unbiased, with and without replacement."""

from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq.errors import SamplingError
from shapiq.sampling._base import (
    Border,
    BorderSizeWithReplacement,
    PairedBorderSizeWithReplacement,
    PairedSizeWithoutReplacement,
    PairedSizeWithReplacement,
    PairedUniformWithoutReplacement,
)


def _numbers(coalitions: jax.Array | np.ndarray) -> np.ndarray:
    rows = np.asarray(coalitions)
    return (rows * (1 << np.arange(rows.shape[-1]))).sum(axis=-1)


def test_paired_size_sampler_is_unbiased_for_an_asymmetric_size_distribution():
    n_players, q = 4, np.array([1.0, 3.0, 1.0, 0.5, 2.0])
    sampler = PairedSizeWithReplacement(keys=jax.random.key(0)[None], weights=jnp.asarray(q))
    n_pairs = 2000
    sample, _ = sampler.sample(2 * n_pairs)
    numbers, log_weights = _numbers(sample.coalitions[0]), np.asarray(sample.log_weights[0])
    complement = 2**n_players - 1 - np.arange(2**n_players)
    pair = np.minimum(np.arange(2**n_players), complement)  # a pair by its smaller bitmask
    weight = dict(zip(pair[numbers].tolist(), np.exp(log_weights).tolist(), strict=False))
    sizes = np.array([x.bit_count() for x in range(2**n_players)])
    p = q[sizes] / q.sum() / np.array([comb(n_players, s) for s in sizes])  # one size draw
    values = np.random.default_rng(0).normal(size=2**n_players)
    # Each of the pair draws hits {x, ~x} with p(x) + p(~x) and adds both weighted values.
    expected = n_pairs * sum(
        (p[x] + p[complement[x]]) * weight[x] * (values[x] + values[complement[x]]) for x in weight
    )

    assert np.array_equal(numbers[1::2], complement[numbers[::2]])  # [S, ~S, ...]
    assert np.array_equal(log_weights[::2], log_weights[1::2])  # both members share the weight
    assert len(weight) == 2 ** (n_players - 1)
    assert expected == pytest.approx(values.sum(), rel=1e-5)


@pytest.mark.parametrize("cls", [BorderSizeWithReplacement, PairedBorderSizeWithReplacement])
def test_border_samplers_enumerate_the_filled_sizes_once_and_stay_unbiased(cls):
    n_players, budget, n_runs = 6, 40, 2000
    # Sizes 1 and 5 would get about 10.5 draws but hold 6 coalitions each: they are enumerated.
    q = jnp.array([0.0, 1 / 5, 1 / 8, 1 / 9, 1 / 8, 1 / 5, 0.0])
    sampler = cls(keys=jax.random.split(jax.random.key(3), n_runs), weights=q)
    sample, _ = sampler.sample(budget)
    numbers, weights = _numbers(sample.coalitions), np.exp(np.asarray(sample.log_weights))
    sizes = np.asarray(sample.coalitions.sum(axis=-1))
    border = (sizes == 1) | (sizes == n_players - 1)
    every_border = [x for x in range(2**n_players) if x.bit_count() in (1, n_players - 1)]
    values = np.random.default_rng(3).normal(size=2**n_players)
    values[[0, -1]] = 0.0  # q gives the empty and grand coalitions no weight
    estimates = np.sum(weights * values[numbers], axis=-1)  # (n_runs,)
    standard_error = np.std(estimates) / np.sqrt(n_runs)

    assert all(sorted(row[mask]) == every_border for row, mask in zip(numbers, border, strict=True))
    assert np.allclose(weights[border], 1.0)  # enumerated coalitions count once
    assert isinstance(sampler, Border)
    assert abs(estimates.mean() - values.sum()) < 4 * standard_error


@pytest.mark.parametrize(
    ("make", "n_players"),
    [
        (lambda keys, n: PairedUniformWithoutReplacement(keys=keys, n_players=n), 4),
        (
            lambda keys, n: PairedSizeWithoutReplacement(keys=keys, weights=jnp.ones(n + 1)),
            4,
        ),  # the middle size halves
        (lambda keys, n: PairedSizeWithoutReplacement(keys=keys, weights=jnp.ones(n + 1)), 5),
    ],
)
def test_paired_samplers_without_replacement_draw_every_pair_once_and_then_refuse(make, n_players):
    sampler = make(jax.random.split(jax.random.key(1), 2), n_players)
    samples = []
    for budget in (2, 4, 6, 2**n_players - 12):
        sample, sampler = sampler.sample(budget)
        samples.append(sample)
    numbers = _numbers(np.concatenate([sample.coalitions for sample in samples], axis=-2))
    values = np.random.default_rng(1).normal(size=2**n_players)
    estimate = np.sum(np.exp(samples[-1].log_weights[0, 0]) * values[numbers[0]])

    assert all(sorted(row) == list(range(2**n_players)) for row in numbers)
    assert estimate == pytest.approx(values.sum(), rel=1e-5)  # every pair drawn: exact
    assert isinstance(sampler, Border)  # full strata are enumerated without asking
    with pytest.raises(SamplingError):
        sampler.sample(2)
    with pytest.raises(ValueError, match="split a pair"):
        sampler.sample(1)
