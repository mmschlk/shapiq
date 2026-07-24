"""Tests for exact projection: the tower, fidelity, and the read-outs."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    CallableGame,
    FourierBasis,
    MoebiusBasis,
    banzhaf_values,
    fidelity,
    project,
    shapley_values,
    soft_shapley_measure,
    to_basis,
    uniform_measure,
)
from shapiq.coalitions import DenseCoalitionArray
from shapiq.games import all_coalitions, interaction_terms

N_PLAYERS = 8


def generic_game(seed: int = 3) -> CallableGame:
    """A random full table: structure at every order, so projections are lossy."""
    table = np.random.default_rng(seed).normal(size=2**N_PLAYERS)
    weights = jnp.asarray(1 << np.arange(N_PLAYERS), dtype=jnp.int32)

    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.int32)
        return jnp.asarray(table, dtype=jnp.float32)[masks @ weights]

    return CallableGame(fn=fn, n_players=N_PLAYERS)


def test_the_projection_tower_composes_under_a_shared_measure():
    game = generic_game()
    uniform = uniform_measure(N_PLAYERS)
    # the intermediate projection must be genuinely lossy or the test is vacuous
    assert fidelity(game, project(game, 2, uniform), uniform) < 0.5
    for measure in (uniform, soft_shapley_measure(N_PLAYERS)):
        two_step = project(project(game, 2, measure), 1, measure)
        one_step = project(game, 1, measure)
        for term in interaction_terms(N_PLAYERS, 1):
            assert two_step[tuple(term)] == pytest.approx(one_step[tuple(term)], abs=1e-8)


def test_mixed_measures_break_the_tower():
    game = generic_game()
    mixed = project(project(game, 2, uniform_measure(N_PLAYERS)), 1, soft_shapley_measure(N_PLAYERS))
    direct = project(game, 1, soft_shapley_measure(N_PLAYERS))
    gap = max(
        abs(mixed[tuple(term)] - direct[tuple(term)])
        for term in interaction_terms(N_PLAYERS, 1)
    )
    assert gap > 0.05  # the measure is part of the index


def test_fidelity_climbs_the_order_dial_to_exactness():
    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense())[..., :4]
        return masks.any(axis=-1).astype(jnp.float32)

    or_game = CallableGame(fn=fn, n_players=N_PLAYERS)
    uniform = uniform_measure(N_PLAYERS)
    scores = [fidelity(or_game, project(or_game, order, uniform), uniform) for order in range(5)]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.0, abs=1e-9)
    assert scores[4] == pytest.approx(1.0, abs=1e-9)


def test_read_outs_match_the_full_table_oracles():
    game = generic_game(seed=11)
    exact = to_basis(game, MoebiusBasis())
    masks = all_coalitions(N_PLAYERS)
    table = np.asarray(game(DenseCoalitionArray(jnp.asarray(masks))), dtype=np.float64)
    row_weights = 1 << np.arange(N_PLAYERS)

    def index_of(mask: np.ndarray) -> int:
        return int((mask * row_weights).sum())


    sv_oracle = np.zeros(N_PLAYERS)
    bv_oracle = np.zeros(N_PLAYERS)
    for i in range(N_PLAYERS):
        for mask in masks[~masks[:, i]]:
            with_i = mask.copy()
            with_i[i] = True
            delta = table[index_of(with_i)] - table[index_of(mask)]
            size = int(mask.sum())
            sv_oracle[i] += (
                math.factorial(size) * math.factorial(N_PLAYERS - size - 1)
                / math.factorial(N_PLAYERS) * delta
            )
            bv_oracle[i] += delta / 2 ** (N_PLAYERS - 1)
    assert np.allclose(shapley_values(exact), sv_oracle, atol=1e-6)
    assert np.allclose(banzhaf_values(exact), bv_oracle, atol=1e-6)
    # the Fourier bridge: Banzhaf values are twice the degree-one coefficients
    fourier = to_basis(game, FourierBasis())
    fourier_singles = np.array([2.0 * fourier[(i,)] for i in range(N_PLAYERS)])
    assert np.allclose(fourier_singles, bv_oracle, atol=1e-6)
