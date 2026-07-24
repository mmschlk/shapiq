"""Tests for parametric games: bases, the two planes, and game algebra."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import CallableGame, ParametricGame, shapley_values, to_basis
from shapiq.coalitions import DenseCoalitionArray
from shapiq.games import all_coalitions

N_PLAYERS = 8
BLOCK = (0, 1, 2, 3)


def logic_game(kind: str):
    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense())[..., BLOCK]
        if kind == "and":
            return masks.all(axis=-1).astype(jnp.float32)
        if kind == "or":
            return masks.any(axis=-1).astype(jnp.float32)
        return (masks.sum(axis=-1) % 2).astype(jnp.float32)

    return CallableGame(fn=fn, n_players=N_PLAYERS)


def test_sparsity_is_basis_relative():
    # the R1 matrix: each logic atom is sparse exactly in its own basis
    expected = {
        "and": {"moebius": 1, "comoebius": 15, "fourier": 16},
        "or": {"moebius": 15, "comoebius": 1, "fourier": 16},
        "xor": {"moebius": 15, "comoebius": 15, "fourier": 2},
    }
    for kind, per_basis in expected.items():
        game = logic_game(kind)
        for basis, support_size in per_basis.items():
            assert len(to_basis(game, basis).support) == support_size, (kind, basis)


def test_the_or_game_smears_by_inclusion_exclusion_in_moebius():
    exact = to_basis(logic_game("or"), "moebius")
    for size in (1, 2, 3, 4):
        for members in [tuple(BLOCK[:size])]:
            assert exact[members] == pytest.approx((-1.0) ** (size + 1), abs=1e-9)
    or_native = to_basis(logic_game("or"), "comoebius")
    assert or_native.support == (frozenset(BLOCK),)
    assert or_native[BLOCK] == pytest.approx(1.0, abs=1e-9)


def test_the_two_planes_are_distinct_verbs():
    game = ParametricGame("moebius", {(): 1.0, (0,): 2.0, (0, 1): -1.0}, 3)
    # coefficient plane: read-outs, zero for absent terms
    assert game[(0,)] == 2.0
    assert game[(2,)] == 0.0
    # game plane: evaluation includes the intercept and lower orders
    masks = jnp.asarray([[True, True, False], [False, False, False]])
    values = game(DenseCoalitionArray(masks))
    assert np.allclose(np.asarray(values), [2.0, 1.0])


def test_game_algebra_is_pointwise():
    left = ParametricGame("moebius", {(0,): 1.0, (0, 1): 0.5}, 4)
    right = ParametricGame("moebius", {(1,): -2.0}, 4)
    coalitions = DenseCoalitionArray(jnp.asarray(all_coalitions(4)))
    combined = np.asarray((left - 2.0 * right)(coalitions))
    expected = np.asarray(left(coalitions)) - 2.0 * np.asarray(right(coalitions))
    assert np.allclose(combined, expected, atol=1e-6)


def test_teaching_errors_name_the_fix():
    with pytest.raises(ValueError, match="shipped bases are"):
        ParametricGame("walsh", {}, 3)
    with pytest.raises(ValueError, match=r"outside 0\.\.2"):
        ParametricGame("moebius", {(3,): 1.0}, 3)
    fourier_game = ParametricGame("fourier", {(0,): 1.0}, 3)
    with pytest.raises(ValueError, match="to_basis"):
        shapley_values(fourier_game)


def test_exact_change_of_basis_roundtrips():
    rng = np.random.default_rng(0)
    truth = ParametricGame(
        "moebius",
        {members: float(rng.normal()) for members in [(), (0,), (2,), (0, 1), (1, 2, 3)]},
        5,
    )
    fourier = to_basis(truth, "fourier")
    back = to_basis(fourier, "moebius")
    for term in truth.terms:
        assert back[tuple(term)] == pytest.approx(truth[tuple(term)], abs=1e-9)
