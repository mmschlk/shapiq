"""Discrete derivative games keep a shared game compact across reference coalitions."""

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
from shapiq.game.discretederivative import DiscreteDerivativeInteractionIndex
from shapiq.game.weights import DDBanzhafWeight, DDShapleyWeight, DDWeightedBanzhafWeight
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


def _banzhaf(size):
    return 1 / 2 ** (N_PLAYERS - 2)


def _index(coefficients, reference_coalition, weights):
    return DiscreteDerivativeInteractionIndex(
        game=_CubicGame(coefficients, FinitePlayers(N_PLAYERS)),
        weights=weights,
        reference_coalition=reference_coalition,
        order=2,
    )


def _expected(coalitions, closed_form):
    return jnp.array(
        [
            [
                0.0
                if (c[i] | c[j])
                else closed_form(int(c.sum())) * _derivative(COEFFICIENTS, c, i, j)
                for i, j in PAIRS
            ]
            for c in coalitions
        ]
    )


@pytest.mark.parametrize(
    ("weights", "closed_form"),
    [
        (DDShapleyWeight(), lambda size: 1 / ((N_PLAYERS - 1) * comb(N_PLAYERS - 2, size))),
        (DDBanzhafWeight(), _banzhaf),
        (
            DDWeightedBanzhafWeight(jnp.array(0.3)),
            lambda size: 0.3**size * 0.7 ** (N_PLAYERS - 2 - size),
        ),
    ],
    ids=["shapley", "banzhaf", "weighted-banzhaf"],
)
@pytest.mark.parametrize("packed_query", [False, True], ids=["bool-query", "packed-query"])
@pytest.mark.parametrize(
    "packed_reference", [False, True], ids=["bool-reference", "packed-reference"]
)
def test_weighted_derivatives_for_bool_and_packed_coalitions(
    weights, closed_form, packed_reference, packed_query
):
    reference = jnp.stack([_coalition(*pair) for pair in PAIRS])
    index = _index(
        COEFFICIENTS, pack_bool_coalition(reference) if packed_reference else reference, weights
    )
    coalitions = jnp.stack([_coalition(), _coalition(3), _coalition(0), _coalition(0, 3)])

    query = pack_bool_coalition(coalitions) if packed_query else coalitions
    assert jnp.allclose(index(query[:, None, :]), _expected(coalitions, closed_form))


def test_shared_game_stays_compact_under_basic_indexing():
    index = _index(
        COEFFICIENTS, jnp.stack([_coalition(*pair) for pair in PAIRS]), DDBanzhafWeight()
    )
    coalitions = jnp.stack([_coalition(3), _coalition(0), _coalition()])[:, None, :]
    expected = _expected(coalitions[:, 0], _banzhaf)

    assert index.shape == (3,)
    assert index.game.shape == ()
    assert jnp.allclose(index(coalitions), expected)
    for result, evaluated, expected_result in [
        (index[1], index[1](coalitions), expected[:, 1:2]),
        (index[1:], index[1:](coalitions), expected[:, 1:]),
        (index.reshape(3, 1), index.reshape(3, 1)(coalitions[:, None]), expected[..., None]),
    ]:
        assert result.game.shape == ()
        assert jnp.allclose(evaluated, expected_result)


def test_game_batch_broadcasts_a_single_reference_coalition():
    coefficients = jnp.stack([COEFFICIENTS, -COEFFICIENTS[::-1]])
    index = _index(coefficients, _coalition(0, 1), DDBanzhafWeight())

    assert index.shape == (2,)
    assert index.reference_coalition.shape == (2, N_PLAYERS)
    assert jnp.allclose(
        index(_coalition(2)),
        jnp.array([_banzhaf(1) * _derivative(c, _coalition(2), 0, 1) for c in coefficients]),
    )
