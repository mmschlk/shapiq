"""Discrete derivative games keep a shared game compact across interactions."""

from dataclasses import dataclass
from math import comb
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, Shaped
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import BoolFinitePlayerGame
from shapiq.game.discretederivative import DiscreteDerivative, powerset, subset_patterns
from shapiq.players import FinitePlayers

N_PLAYERS = 4
PAIRS = [(0, 1), (1, 2), (2, 3)]
COEFFICIENTS = jnp.array([1.0, 2.0, 3.0, 4.0])


@dataclass
class _CubicGame(BoolFinitePlayerGame[Any, jax.Array], JaxAxisProtected[jax.Array]):
    """v(S) = (sum of the coefficients in S)**3; cubic so that pair derivatives depend on S."""

    coefficients: Float[jax.Array, "*batch n_players"]
    _players: Shaped[FinitePlayers, "#*batch"]
    protected_axes: ClassVar[dict[str, int]] = {"coefficients": 1}

    @classmethod
    def _pytree_child_names(cls):
        return ("coefficients", "_players")

    @property
    def players(self) -> FinitePlayers:
        return self._players

    def _eval_bool(self, coalition):
        return jnp.sum(coalition * self.coefficients, axis=-1) ** 3


def _coalition(*players):
    return jnp.isin(jnp.arange(N_PLAYERS), jnp.array(players, dtype=int))


def _derivative(coefficients, coalition, i, j):
    def v(*extra):
        return jnp.sum((coalition | _coalition(*extra)) * coefficients) ** 3

    return v(i, j) - v(i) - v(j) + v()


def _pair_derivatives(coefficients, coalitions):
    return jnp.array([[_derivative(coefficients, c, i, j) for i, j in PAIRS] for c in coalitions])


def _pairs_derivative(coefficients=COEFFICIENTS, interaction=None):
    interaction = (
        jnp.stack([_coalition(*pair) for pair in PAIRS]) if interaction is None else interaction
    )
    game = _CubicGame(coefficients, FinitePlayers(N_PLAYERS))
    return DiscreteDerivative(game=game, interaction=interaction, order=2)


@pytest.mark.parametrize("packed", [False, True], ids=["bool", "packed"])
def test_discrete_derivative_is_the_pair_derivative(packed):
    interaction = jnp.stack([_coalition(*pair) for pair in PAIRS])
    derivative = _pairs_derivative(
        interaction=pack_bool_coalition(interaction) if packed else interaction
    )
    coalitions = subset_patterns(N_PLAYERS)

    assert jnp.allclose(
        derivative(coalitions[:, None, :]), _pair_derivatives(COEFFICIENTS, coalitions), rtol=1e-5
    )


def test_shared_game_stays_compact_under_basic_indexing():
    derivative = _pairs_derivative()
    coalitions = jnp.stack([_coalition(3), _coalition(0), _coalition()])
    expected = _pair_derivatives(COEFFICIENTS, coalitions)
    query = coalitions[:, None, :]

    assert derivative.shape == (3,)
    assert derivative.game.shape == ()
    for result, evaluated, expected_result in [
        (derivative[1], derivative[1](query), expected[:, 1:2]),
        (derivative[1:], derivative[1:](query), expected[:, 1:]),
        (derivative.reshape(3, 1), derivative.reshape(3, 1)(query[:, None]), expected[..., None]),
    ]:
        assert result.game.shape == ()
        assert jnp.allclose(evaluated, expected_result, rtol=1e-5)


def test_game_batch_broadcasts_a_single_interaction():
    coefficients = jnp.stack([COEFFICIENTS, -COEFFICIENTS[::-1]])
    derivative = _pairs_derivative(coefficients, _coalition(0, 1))

    assert derivative.shape == (2,)
    assert derivative.interaction.shape == (2, N_PLAYERS)
    assert jnp.allclose(
        derivative(_coalition(2)),
        jnp.array([_derivative(c, _coalition(2), 0, 1) for c in coefficients]),
    )


def test_powerset_size_filter_is_static_under_jit():
    """Size-filtered subsets keep powerset order, beyond eight players, inside jit."""
    coalition = jnp.array([True, False, True, True, False, True, True, True, True, True])
    full = powerset(coalition, order=8)
    sizes = jnp.sum(full, axis=-1)
    expected = full[(sizes >= 2) & (sizes <= 3)]
    filtered = jax.jit(lambda c: powerset(c, order=8, min_size=2, max_size=3))(coalition)
    assert filtered.shape == (comb(8, 2) + comb(8, 3), 10)
    assert jnp.array_equal(filtered, expected)
    assert subset_patterns(9).shape == (2**9, 9)
    assert jnp.array_equal(
        jnp.sum(subset_patterns(9), axis=-1), jnp.array([i.bit_count() for i in range(512)])
    )
