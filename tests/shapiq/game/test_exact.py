"""The exact computer matches the index definitions and broadcasts game and index batches."""

from itertools import chain, combinations
from math import exp, factorial, lgamma

import jax
import jax.numpy as jnp
import pytest

from shapiq.game.exact import ExactComputer
from shapiq.game.moebius import AdditiveMoebiusGame, SetMoebiusGame
from shapiq.game.table import TableGame
from shapiq.index import (
    AlphaBetaIndex,
    BanzhafIndex,
    ShapleyInteractionIndex,
    ShapleyValueIndex,
    WeightedBanzhafIndex,
)
from shapiq.players import FinitePlayers

N_PLAYERS = 4
TABLES = jax.random.normal(jax.random.key(0), (2, 2**N_PLAYERS))


def _sii(s: int, t: int, n: int) -> float:
    return factorial(s) * factorial(n - s - t) / factorial(n - t + 1)


def _bii(s: int, t: int, n: int) -> float:
    del s
    return 0.5 ** (n - t)


def _beta_4_1(s: int, t: int, n: int) -> float:
    # B(s + beta, n - t - s + alpha) / B(alpha, beta) with alpha = 4, beta = 1.
    def log_beta(a: float, b: float) -> float:
        return lgamma(a) + lgamma(b) - lgamma(a + b)

    return exp(log_beta(s + 1.0, n - t - s + 4.0) - log_beta(4.0, 1.0))


def _brute_force(table: list[float], weight, interactions: list[tuple[int, ...]]) -> list[float]:
    """phi(T) = sum over S outside T of w(|S|, |T|, n) * sum over L ⊆ T of (-1)**|T-L| v(S | L)."""

    def value(players: tuple[int, ...]) -> float:
        return table[sum(1 << i for i in players)]

    result = []
    for interaction in interactions:
        rest = [i for i in range(N_PLAYERS) if i not in interaction]
        subsets = chain.from_iterable(combinations(rest, r) for r in range(len(rest) + 1))
        result.append(
            sum(
                weight(len(coalition), len(interaction), N_PLAYERS)
                * sum(
                    (-1) ** (len(interaction) - r) * value(coalition + part)
                    for r in range(len(interaction) + 1)
                    for part in combinations(interaction, r)
                )
                for coalition in subsets
            )
        )
    return result


@pytest.mark.parametrize(
    ("index", "weight"),
    [
        (ShapleyValueIndex(), _sii),
        (BanzhafIndex(max_order=1), _bii),
        (ShapleyInteractionIndex(max_order=2), _sii),
        (BanzhafIndex(max_order=3), _bii),
        (AlphaBetaIndex(alpha=4.0, beta=1.0, max_order=2), _beta_4_1),
    ],
)
def test_exact_matches_the_index_definition(index, weight):
    computer = ExactComputer.from_game(TableGame(TABLES, FinitePlayers(N_PLAYERS)))
    result = computer(index)
    sets = [tuple(int(i) for i in jnp.flatnonzero(row)) for row in index.interactions(N_PLAYERS)]
    expected = jnp.array([_brute_force(table.tolist(), weight, sets) for table in TABLES])

    assert isinstance(result, AdditiveMoebiusGame | SetMoebiusGame)
    assert jnp.allclose(result.values, expected, atol=1e-5)


def test_game_and_index_batches_broadcast_and_the_computer_runs_under_jit():
    computer = ExactComputer.from_game(TableGame(TABLES, FinitePlayers(N_PLAYERS)))
    p_values = jnp.array([0.3, 0.5, 0.7])
    index = WeightedBanzhafIndex(p=p_values, max_order=2)

    every_pair = computer[:, None](index)  # (games, p values)
    jitted = jax.jit(lambda c: c(ShapleyValueIndex()).values)(computer)
    single = ExactComputer.from_game(TableGame(TABLES[1], FinitePlayers(N_PLAYERS)))
    single_p = WeightedBanzhafIndex(p=p_values[2], max_order=2)

    assert every_pair.shape == (2, 3)
    assert jnp.allclose(every_pair[1, 2].values, single(single_p).values, atol=1e-6)
    assert jnp.allclose(jitted, computer(ShapleyValueIndex()).values, atol=1e-5)
