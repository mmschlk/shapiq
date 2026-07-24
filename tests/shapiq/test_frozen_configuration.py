"""Tests pinning that the configuration tier is frozen, not just documented so."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    SV,
    BasisGame,
    CallableGame,
    ExactExplainer,
    MoebiusBasis,
    Regression,
    ShapleyKernelSampler,
)

N_PLAYERS = 4


def additive_game():
    return CallableGame(
        fn=lambda c: jnp.sum(jnp.asarray(c.to_dense(), dtype=jnp.float32), axis=-1),
        n_players=N_PLAYERS,
    )


def test_policies_are_frozen_after_construction():
    policy = Regression(additive_game(), SV(), random_state=0)
    with pytest.raises(AttributeError, match="frozen"):
        policy.deduplicate = True
    with pytest.raises(AttributeError, match="frozen"):
        policy.brand_new_attribute = 1


def test_games_and_samplers_are_frozen_after_construction():
    game = additive_game()
    # dataclass games guard with FrozenInstanceError, plain games with the
    # teaching error; both are AttributeErrors and both refuse
    with pytest.raises(AttributeError, match=r"frozen|cannot assign"):
        game.n_players = 7
    sampler = ShapleyKernelSampler(N_PLAYERS)
    with pytest.raises(AttributeError, match="frozen"):
        sampler.share_samples = True


def test_basis_games_own_immutable_coefficients():
    game = BasisGame(MoebiusBasis(), {frozenset([0]): 1.5}, N_PLAYERS)
    with pytest.raises(ValueError, match="read-only"):
        game.coefficients[..., 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        game._values[..., 0] = 0.0
    # construction copies: mutating the caller's array cannot retype the game
    source = np.ones(3)
    fitted = BasisGame(
        MoebiusBasis(),
        None,
        N_PLAYERS,
        terms=(frozenset([0]), frozenset([1]), frozenset([2])),
        values=source,
    )
    source[0] = 99.0
    assert fitted[(0,)] == 1.0


def test_lazy_caches_still_work_on_frozen_explainers():
    explainer = ExactExplainer(additive_game(), SV())
    first = explainer.estimate()
    second = explainer.estimate()
    assert float(first[(0,)]) == float(second[(0,)]) == pytest.approx(1.0)
