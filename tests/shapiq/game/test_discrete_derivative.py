"""Discrete derivative games keep a shared game compact across interactions."""

from dataclasses import dataclass
from math import comb, exp, lgamma
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, Shaped
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import BoolFinitePlayerGame
from shapiq.game.discretederivative import (
    DiscreteDerivativeInteractionIndex,
    WeightedGameInteractionIndex,
    subset_patterns,
)
from shapiq.game.weights import (
    DDAlphaBetaWeight,
    DDBanzhafWeight,
    DDShapleyWeight,
    DDWeightedBanzhafWeight,
)
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


def _shapley(size):
    return 1 / ((N_PLAYERS - 1) * comb(N_PLAYERS - 2, size))


def _weighted_banzhaf(p):
    return lambda size: p**size * (1 - p) ** (N_PLAYERS - 2 - size)


def _beta(alpha, beta):
    def log_beta(a, b):
        return lgamma(a) + lgamma(b) - lgamma(a + b)

    return lambda size: exp(
        log_beta(size + beta, N_PLAYERS - 2 - size + alpha) - log_beta(alpha, beta)
    )


# Each case: weights (scalar or batched along a leading axis) and one closed form per batch entry.
WEIGHT_CASES = {
    "shapley": (DDShapleyWeight(), [_shapley]),
    "banzhaf": (DDBanzhafWeight(), [_banzhaf]),
    "weighted-banzhaf-batch": (
        DDWeightedBanzhafWeight(jnp.array([0.3, 0.5, 0.9])[:, None]),
        [_weighted_banzhaf(p) for p in (0.3, 0.5, 0.9)],
    ),
    "beta-batch": (
        DDAlphaBetaWeight(jnp.array([1.0, 4.0, 1.0])[:, None], jnp.array([1.0, 1.0, 4.0])[:, None]),
        [_shapley, _beta(4.0, 1.0), _beta(1.0, 4.0)],
    ),
}


def _query(coalitions, index):
    """Give the coalitions one size-1 axis per index axis, so they broadcast against the index batch."""
    return coalitions.reshape(coalitions.shape[0], *(1,) * index.ndim, coalitions.shape[-1])


def _index(coefficients, interaction, weights):
    return DiscreteDerivativeInteractionIndex(
        game=_CubicGame(coefficients, FinitePlayers(N_PLAYERS)),
        weights=weights,
        interaction=interaction,
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


def _expected_batch(coalitions, weights, closed_forms):
    """Stack the expectation of every weight batch entry; (coalitions, *weight_batch, interactions)."""
    expected = jnp.stack([_expected(coalitions, form) for form in closed_forms], axis=1)
    return expected[:, 0] if weights.ndim == 0 else expected


@pytest.mark.parametrize(
    ("weights", "closed_forms"), WEIGHT_CASES.values(), ids=WEIGHT_CASES.keys()
)
@pytest.mark.parametrize("packed_query", [False, True], ids=["bool-query", "packed-query"])
@pytest.mark.parametrize(
    "packed_interaction", [False, True], ids=["bool-interaction", "packed-interaction"]
)
def test_weighted_derivatives_for_bool_and_packed_coalitions(
    weights, closed_forms, packed_interaction, packed_query
):
    interaction = jnp.stack([_coalition(*pair) for pair in PAIRS])
    index = _index(
        COEFFICIENTS,
        pack_bool_coalition(interaction) if packed_interaction else interaction,
        weights,
    )
    coalitions = jnp.stack([_coalition(), _coalition(3), _coalition(0), _coalition(0, 3)])
    query = pack_bool_coalition(coalitions) if packed_query else coalitions

    assert index.shape == (*weights.shape[:-1], len(PAIRS))
    assert jnp.allclose(
        index(_query(query, index)), _expected_batch(coalitions, weights, closed_forms)
    )


def test_shared_game_stays_compact_under_basic_indexing():
    index = _index(
        COEFFICIENTS, jnp.stack([_coalition(*pair) for pair in PAIRS]), DDBanzhafWeight()
    )
    coalitions = jnp.stack([_coalition(3), _coalition(0), _coalition()])[:, None, :]
    expected = _expected(coalitions[:, 0], _banzhaf)

    assert index.shape == (3,)
    assert index.game.shape == ()
    assert index.weights.shape == ()
    assert jnp.allclose(index(coalitions), expected)
    for result, evaluated, expected_result in [
        (index[1], index[1](coalitions), expected[:, 1:2]),
        (index[1:], index[1:](coalitions), expected[:, 1:]),
        (index.reshape(3, 1), index.reshape(3, 1)(coalitions[:, None]), expected[..., None]),
    ]:
        assert result.game.shape == ()
        assert result.weights.shape == ()
        assert jnp.allclose(evaluated, expected_result)


def test_game_batch_broadcasts_a_single_interaction():
    coefficients = jnp.stack([COEFFICIENTS, -COEFFICIENTS[::-1]])
    index = _index(coefficients, _coalition(0, 1), DDBanzhafWeight())

    assert index.shape == (2,)
    assert index.interaction.shape == (2, N_PLAYERS)
    assert jnp.allclose(
        index(_coalition(2)),
        jnp.array([_banzhaf(1) * _derivative(c, _coalition(2), 0, 1) for c in coefficients]),
    )


@pytest.mark.parametrize(
    "weights", [case[0] for case in WEIGHT_CASES.values()], ids=WEIGHT_CASES.keys()
)
@pytest.mark.parametrize("packed", [False, True], ids=["bool", "packed"])
def test_weighted_game_index_sums_to_the_discrete_derivative_index(weights, packed):
    interaction = jnp.stack([_coalition(*pair) for pair in PAIRS])
    game = _CubicGame(COEFFICIENTS, FinitePlayers(N_PLAYERS))
    weighted = WeightedGameInteractionIndex(
        game=game,
        weights=weights,
        interaction=pack_bool_coalition(interaction) if packed else interaction,
    )
    derivative = _index(COEFFICIENTS, interaction, weights)
    all_coalitions = subset_patterns(N_PLAYERS)

    assert weighted.shape == derivative.shape == (*weights.shape[:-1], len(PAIRS))
    assert weighted.game.shape == ()
    assert jnp.allclose(
        jnp.sum(weighted(_query(all_coalitions, weighted)), axis=0),
        jnp.sum(derivative(_query(all_coalitions, derivative)), axis=0),
    )


@pytest.mark.parametrize(
    "index_class", [DiscreteDerivativeInteractionIndex, WeightedGameInteractionIndex]
)
def test_index_is_constructible_and_batchable_inside_jit(index_class):
    interactions = jnp.stack([_coalition(*pair) for pair in PAIRS])
    game = _CubicGame(COEFFICIENTS, FinitePlayers(N_PLAYERS))
    extra = {"order": 2} if index_class is DiscreteDerivativeInteractionIndex else {}
    all_coalitions = subset_patterns(N_PLAYERS)

    def total(p):
        index = index_class(
            game=game, weights=DDWeightedBanzhafWeight(p), interaction=interactions, **extra
        )
        return jnp.sum(index(_query(all_coalitions, index)), axis=0)

    ps = jnp.array([0.3, 0.5, 0.7])
    eager = jnp.stack([total(float(p)) for p in ps])

    # The p-batch inside the index and vmap over scalar p give the same result.
    assert jnp.allclose(jax.jit(total)(ps[:, None]), eager)
    assert jnp.allclose(jax.jit(jax.vmap(total))(ps), eager)


@pytest.mark.parametrize(
    "index_class", [DiscreteDerivativeInteractionIndex, WeightedGameInteractionIndex]
)
def test_weight_batch_is_part_of_the_index_batch(index_class):
    interactions = jnp.stack([_coalition(*pair) for pair in PAIRS])
    extra = {"order": 2} if index_class is DiscreteDerivativeInteractionIndex else {}
    ps = jnp.array([0.2, 0.5, 0.9, 0.3])

    def index_for(p):
        return index_class(
            game=_CubicGame(COEFFICIENTS, FinitePlayers(N_PLAYERS)),
            weights=DDWeightedBanzhafWeight(p),
            interaction=interactions,
            **extra,
        )

    index = index_for(ps[:, None])
    coalitions = subset_patterns(N_PLAYERS)[:, None, None, :]

    assert index.shape == (4, 3)
    assert index.weights.shape == (4, 1)
    assert index.game.shape == ()
    for i, p in enumerate(ps):
        assert jnp.allclose(index[i](coalitions[:, 0]), index_for(p)(coalitions[:, 0]))
    assert jnp.allclose(index(coalitions)[:, 2], index_for(ps[2])(coalitions[:, 0]))
