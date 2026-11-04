"""Möbius and Fourier games reproduce the coalition table they were transformed from."""

import jax
import jax.numpy as jnp
import pytest

from shapiq.game.discretederivative import subset_patterns
from shapiq.game.fourier import FourierGame
from shapiq.game.moebius import AdditiveMoebiusGame, MoebiusGame, SetMoebiusGame
from shapiq.game.table import TableGame
from shapiq.players import FinitePlayers

N_PLAYERS = 3
# Rows are coalitions in table order and double as the full list of interactions.
SETS = subset_patterns(N_PLAYERS)
TABLES = jnp.array(
    [[0.0, 1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 10.0], [1.0, -2.0, 0.5, 3.0, 2.0, 0.0, 7.0, -1.0]]
)


def _overlap(left: jax.Array, right: jax.Array) -> jax.Array:
    return jnp.sum(left[:, None, :] & right[None, :, :], axis=-1)


def _moebius(tables: jax.Array) -> SetMoebiusGame:
    # m(T) = sum over S ⊆ T of (-1)**|T \ S| v(S)
    subset = ~jnp.any(SETS[None, :, :] & ~SETS[:, None, :], axis=-1)  # [T, S]: S ⊆ T
    sign = (-1.0) ** (jnp.sum(SETS, axis=-1)[:, None] - _overlap(SETS, SETS))
    return SetMoebiusGame(
        values=tables @ (subset * sign).T,
        interactions=SETS,
        players=FinitePlayers(N_PLAYERS),
        additivity=N_PLAYERS,
    )


def _fourier(tables: jax.Array) -> FourierGame:
    # f(T) = 2**-n sum over S of v(S) (-1)**|T & S|
    chi = (-1.0) ** _overlap(SETS, SETS)
    return FourierGame(
        values=tables @ chi / 2**N_PLAYERS,
        interactions=SETS,
        players=FinitePlayers(N_PLAYERS),
        additivity=N_PLAYERS,
    )


@pytest.mark.parametrize("build", [_moebius, _fourier])
def test_basis_game_reproduces_the_table_batched_indexed_and_jitted(build):
    game = build(TABLES)
    table = TableGame(TABLES, FinitePlayers(N_PLAYERS))
    expected = table(SETS[:, None])  # (coalitions, games)

    assert game.shape == (2,)
    assert jnp.allclose(game(SETS[:, None]), expected, atol=1e-5)
    assert jnp.allclose(game[1](SETS), expected[:, 1], atol=1e-5)
    assert jnp.allclose(jax.jit(lambda g, c: g(c))(game, SETS[:, None]), expected, atol=1e-5)
    assert jax.jit(lambda g: g.additivity)(game) == N_PLAYERS


@pytest.mark.parametrize("game_class", [SetMoebiusGame, FourierGame])
def test_additivity_must_bound_the_interactions(game_class):
    with pytest.raises(ValueError, match="additivity"):
        game_class(
            values=jnp.zeros(len(SETS)),
            interactions=SETS,
            players=FinitePlayers(N_PLAYERS),
            additivity=N_PLAYERS - 1,
        )


def test_additive_moebius_game_is_a_moebius_game_with_implicit_singletons():
    values = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    players = FinitePlayers(N_PLAYERS)
    singletons = jnp.eye(N_PLAYERS, dtype=bool)
    additive = AdditiveMoebiusGame(values=values, players=players)
    stored = SetMoebiusGame(values=values, interactions=singletons, players=players, additivity=1)
    expected = stored(SETS[:, None])

    assert isinstance(additive, MoebiusGame)
    assert additive.additivity == 1
    assert jnp.array_equal(additive.interactions, singletons)
    assert jnp.allclose(additive(SETS[:, None]), expected)
    assert jnp.allclose(additive[1](SETS), expected[:, 1])
    assert jnp.allclose(jax.jit(lambda g, c: g(c))(additive, SETS[:, None]), expected)
