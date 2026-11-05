"""The exact computer matches the index definitions and broadcasts game and index batches."""

from itertools import chain, combinations
from math import exp, factorial, lgamma

import jax
import jax.numpy as jnp
import pytest

from shapiq.game.exact import ExactComputer
from shapiq.game.moebius import AdditiveMoebiusGame, SetMoebiusGame
from shapiq.game.table import TableGame
from shapiq.game.weights import WeightedBanzhafWeight
from shapiq.index import (
    BanzhafInteractionIndex,
    BanzhafValue,
    FaithfulBanzhafTopOrderIndex,
    FaithfulShapleyTopOrderIndex,
    ShapleyInteractionIndex,
    ShapleyValue,
    WeightedBanzhafValue,
)
from shapiq.players import FinitePlayers

N_PLAYERS = 4
TABLES = jax.random.normal(jax.random.key(0), (2, 2**N_PLAYERS))


def _sii(s: int, t: int, n: int) -> float:
    return factorial(s) * factorial(n - s - t) / factorial(n - t + 1)


def _bii(s: int, t: int, n: int) -> float:
    del s
    return 0.5 ** (n - t)


def _fsii_top(s: int, t: int, n: int) -> float:
    # Tsai et al. (2023), Theorem 19, for the top order k = t.
    return exp(lgamma(2 * t) - 2 * lgamma(t) + lgamma(n - s) + lgamma(s + t) - lgamma(n + t))


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
        (ShapleyValue(), _sii),
        (BanzhafValue(), _bii),
        (ShapleyInteractionIndex(max_order=2), _sii),
        (BanzhafInteractionIndex(max_order=3), _bii),
        (FaithfulShapleyTopOrderIndex(max_order=2), _fsii_top),
        (FaithfulBanzhafTopOrderIndex(max_order=3), _bii),
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
    index = WeightedBanzhafValue(weights=WeightedBanzhafWeight(p_values))

    every_pair = computer[:, None](index)  # (games, p values)
    jitted = jax.jit(lambda c: c(ShapleyValue()).values)(computer)
    single = ExactComputer.from_game(TableGame(TABLES[1], FinitePlayers(N_PLAYERS)))
    single_p = WeightedBanzhafValue(weights=WeightedBanzhafWeight(p_values[2]))

    assert every_pair.shape == (2, 3)
    assert jnp.allclose(every_pair[1, 2].values, single(single_p).values, atol=1e-6)
    assert jnp.allclose(jitted, computer(ShapleyValue()).values, atol=1e-5)
