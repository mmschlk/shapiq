"""Swap-or-not permutes every stratum on multi-word integers; the samplers read it without repeats."""

import math
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq.errors import SamplingError
from shapiq.sampling._base import (
    SizeWithoutReplacement,
    UniformWithoutReplacement,
    _huntington_hill,
)
from shapiq.sampling._permutation import (
    colex_unrank,
    from_int,
    from_positions,
    n_rounds,
    n_words,
    random_below,
    swap_or_not,
)


def _to_ints(words: jax.Array) -> list[int]:
    rows = np.asarray(words).reshape(-1, words.shape[-1])
    return [sum(int(word) << (32 * i) for i, word in enumerate(row)) for row in rows]


def _permute(domains: tuple[int, ...], draws: int, width: int) -> list[list[int]]:
    """Permute positions ``0 .. min(N_s, draws) - 1`` of every stratum in one call."""
    rounds = max(n_rounds(n) for n in domains)
    keys = jax.random.split(jax.random.key(0), 2 * len(domains))
    round_keys = jnp.stack(
        [
            jnp.pad(random_below(keys[2 * s], n, (rounds,)), ((0, 0), (0, width - n_words(n))))
            for s, n in enumerate(domains)
        ]
    )
    round_seeds = jnp.stack(
        [jax.random.bits(keys[2 * s + 1], (rounds,), jnp.uint32) for s in range(len(domains))]
    )
    strata = jnp.concatenate([jnp.full(min(n, draws), s) for s, n in enumerate(domains)])
    positions = jnp.concatenate([jnp.arange(min(n, draws)) for n in domains])
    bounds = jnp.asarray(np.stack([from_int(n, width) for n in domains]))[strata]
    images = _to_ints(
        swap_or_not(from_positions(positions, width), bounds, strata, round_keys, round_seeds)
    )
    return [
        [x for x, t in zip(images, strata.tolist(), strict=True) if t == s]
        for s in range(len(domains))
    ]


@pytest.mark.parametrize("extra_words", [0, 2])
def test_swap_or_not_is_a_bijection_of_every_stratum_on_one_and_several_words(extra_words):
    domains = (1, 7, 1000)
    images = _permute(domains, max(domains), n_words(max(domains)) + extra_words)
    assert all(sorted(row) == list(range(n)) for row, n in zip(images, domains, strict=True))


def test_swap_or_not_stays_distinct_and_in_range_beyond_64_bits():
    domains = (comb(80, 40), 2**70)  # about 2**76 and 2**70
    images = _permute(domains, 5000, n_words(max(domains)))
    assert all(
        len(set(row)) == 5000 and max(row) < n for row, n in zip(images, domains, strict=True)
    )


@pytest.mark.parametrize(("n_players", "size"), [(6, 3), (7, 0), (7, 7)])
def test_colex_unrank_inverts_the_colex_rank(n_players, size):
    ranks = from_positions(jnp.arange(comb(n_players, size)), 2)
    coalitions = np.asarray(colex_unrank(ranks, n_players, jnp.asarray(size)))
    colex = [sum(comb(c, i + 1) for i, c in enumerate(np.flatnonzero(row))) for row in coalitions]
    assert (coalitions.sum(axis=-1) == size).all()
    assert colex == list(range(comb(n_players, size)))


def _seat_by_seat(q: list[float], caps: tuple[int, ...], total: int) -> list[int]:
    """Hand out draws one at a time by Huntington-Hill priority, ties to the lowest stratum."""
    seats = [0] * len(q)
    for _ in range(total):
        priority = [
            -1.0 if w <= 0 or k >= c else math.inf if k == 0 else w / math.sqrt(k * (k + 1))
            for w, k, c in zip(q, seats, caps, strict=True)
        ]
        seats[priority.index(max(priority))] += 1
    return seats


@pytest.mark.parametrize(
    ("q", "caps"),
    [
        ([6.0, 6.0, 2.0], (100, 100, 100)),  # largest remainder takes a draw back from 10 to 11
        ([1.0] * 5, (1, 4, 6, 4, 1)),  # exact ties and full strata
        ([0.5, 0.0, 3.0, 0.2, 1.0], (3, 9, 2, 20, 7)),  # an empty stratum and tight capacities
    ],
)
def test_huntington_hill_matches_seat_by_seat_allocation_for_every_total(q, caps):
    allocate = jax.jit(lambda total: _huntington_hill(jnp.asarray(q), caps, total))
    feasible = sum(c for c, w in zip(caps, q, strict=True) if w > 0)
    counts = [np.asarray(allocate(jnp.int32(t))).tolist() for t in range(min(feasible, 60) + 1)]

    assert counts == [_seat_by_seat(q, caps, t) for t in range(len(counts))]
    assert np.all(np.diff(counts, axis=0) >= 0)  # more draws never take one back


def test_uniform_sampler_draws_every_coalition_once_across_steps_and_then_refuses():
    n_players = 4
    sampler = UniformWithoutReplacement(
        keys=jax.random.split(jax.random.key(0), 3), n_players=n_players
    )
    samples = []
    for budget in (5, 6, 5):
        sample, sampler = sampler.sample(budget)
        samples.append(sample)
    coalitions = np.concatenate([sample.coalitions for sample in samples], axis=-2)  # (3, 16, 4)
    numbers = (coalitions * (1 << np.arange(n_players))).sum(axis=-1)
    values = np.random.default_rng(0).normal(size=2**n_players)
    estimate = np.sum(np.exp(samples[-1].log_weights[0, 0]) * values[numbers[0]])

    assert all(sorted(row) == list(range(2**n_players)) for row in numbers)
    assert not np.array_equal(numbers[0], numbers[1])  # independent permutations per key
    assert estimate == pytest.approx(values.sum(), rel=1e-5)  # exhausted: the estimate is exact
    with pytest.raises(SamplingError):
        sampler.sample(1)


def test_size_sampler_covers_every_size_first_and_every_coalition_once_across_steps():
    n_players = 5
    sampler = SizeWithoutReplacement(
        keys=jax.random.split(jax.random.key(1), 2),
        weights=jnp.array([comb(n_players, s) for s in range(n_players + 1)], dtype=float),
    )  # uniform over coalitions
    samples = []
    for budget in (6, 7, 1, 12, 6):
        sample, sampler = sampler.sample(budget)
        samples.append(sample)
    coalitions = np.concatenate([sample.coalitions for sample in samples], axis=-2)  # (2, 32, 5)
    numbers = (coalitions * (1 << np.arange(n_players))).sum(axis=-1)
    values = np.random.default_rng(1).normal(size=2**n_players)
    estimate = np.sum(np.exp(samples[-1].log_weights[0, 0]) * values[numbers[0]])

    assert sorted(samples[0].coalitions[0].sum(axis=-1).tolist()) == list(range(n_players + 1))
    assert all(sorted(row) == list(range(2**n_players)) for row in numbers)
    assert estimate == pytest.approx(values.sum(), rel=1e-5)  # every size exhausted: exact
    with pytest.raises(SamplingError):
        sampler.sample(1)


def test_size_sampler_below_one_draw_per_size_is_random_and_unbiased():
    n_players, n_runs = 8, 4000
    sampler = SizeWithoutReplacement(
        keys=jax.random.split(jax.random.key(2), n_runs), weights=jnp.ones(n_players + 1)
    )
    first, sampler = sampler.sample(3)
    second, _ = sampler.sample(3)  # 6 units, still fewer than the 9 sizes
    coalitions = np.concatenate([first.coalitions, second.coalitions], axis=-2)
    numbers = (coalitions * (1 << np.arange(n_players))).sum(axis=-1)
    sizes = coalitions.sum(axis=-1)
    # Before every size has a draw, E[M_s] = 6/9 units: earlier draws take the latest weight.
    latest = np.array([comb(n_players, s) for s in range(n_players + 1)]) * 9 / 6
    values = np.random.default_rng(2).normal(size=2**n_players)
    estimates = np.sum(latest[sizes] * values[numbers], axis=-1)  # (n_runs,)
    standard_error = np.std(estimates) / np.sqrt(n_runs)

    assert all(len(set(row)) == 6 for row in sizes)  # one draw per size, sizes vary by key
    assert len({tuple(sorted(row)) for row in sizes}) > 1
    assert np.allclose(np.exp(second.log_weights), latest[second.coalitions.sum(axis=-1)])
    assert abs(estimates.mean() - values.sum()) < 4 * standard_error
