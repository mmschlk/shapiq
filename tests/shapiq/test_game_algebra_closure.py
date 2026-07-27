"""Tests pinning that basis games are closed under their algebra."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    SII,
    SV,
    BasisGame,
    CallableGame,
    CoMoebiusBasis,
    Estimate,
    ExactExplainer,
    MoebiusBasis,
    NoEvidence,
    Provenance,
    SumGame,
    shapley_values,
)

N_PLAYERS = 4


def additive_game():
    return CallableGame(
        fn=lambda c: jnp.sum(jnp.asarray(c.to_dense(), dtype=jnp.float32), axis=-1),
        n_players=N_PLAYERS,
    )


def test_same_basis_addition_stays_readable():
    left = BasisGame(MoebiusBasis(), {frozenset(): 1.0, frozenset([0]): 2.0}, N_PLAYERS)
    right = BasisGame(MoebiusBasis(), {frozenset([0]): 0.5, frozenset([1, 2]): -1.0}, N_PLAYERS)
    total = left + right
    assert isinstance(total, BasisGame)
    assert total[(0,)] == 2.5
    assert total[()] == 1.0  # the empty slots add
    assert total[(1, 2)] == -1.0
    residual = total - right
    assert isinstance(residual, BasisGame)
    assert residual[(1, 2)] == pytest.approx(0.0)


def test_mismatched_bases_fall_back_to_the_extensional_sum():
    conjunctive = BasisGame(MoebiusBasis(), {frozenset([0]): 1.0}, N_PLAYERS)
    disjunctive = BasisGame(CoMoebiusBasis(), {frozenset([0]): 1.0}, N_PLAYERS)
    total = conjunctive + disjunctive
    assert isinstance(total, SumGame)
    masks = jnp.asarray(np.eye(N_PLAYERS, dtype=bool))
    assert bool(
        jnp.allclose(total(masks), conjunctive(masks) + disjunctive(masks), atol=1e-6),
    )


def test_scaling_stays_readable():
    game = BasisGame(MoebiusBasis(), {frozenset([1]): 3.0}, N_PLAYERS)
    assert isinstance(2.0 * game, BasisGame)
    assert (2.0 * game)[(1,)] == 6.0


def test_shapley_values_are_linear_over_closed_addition():
    game = additive_game()
    proxy = BasisGame(MoebiusBasis(), {frozenset([p]): 0.5 for p in range(N_PLAYERS)}, N_PLAYERS)
    estimate = ExactExplainer(game, SV()).estimate()
    combined = proxy + estimate
    assert isinstance(combined, BasisGame)
    assert not isinstance(combined, Estimate)  # provenance does not survive addition
    assert np.allclose(
        shapley_values(combined),
        shapley_values(proxy) + shapley_values(estimate.detach()),
        atol=1e-12,
    )


def test_reads_above_the_declared_order_teach_instead_of_zero():
    estimate = ExactExplainer(additive_game(), SV()).estimate()
    with pytest.raises(ValueError, match=r"outside its[\s\S]*space"):
        estimate[(0, 1)]
    # within the declared order, absent terms stay sparsity zeros
    sparse = Estimate(
        BasisGame(MoebiusBasis(), {frozenset([0]): 1.0}, N_PLAYERS),
        Provenance(evidence=NoEvidence(), index=SII(order=2)),
    )
    assert sparse[(0, 1)] == 0.0
