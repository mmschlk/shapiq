"""Tests pinning that the configuration tier is frozen, not just documented so."""

from __future__ import annotations

import copy
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    SV,
    BasisGame,
    CallableGame,
    ExactExplainer,
    InterventionalTreeGame,
    MoebiusBasis,
    Regression,
    ShapleyKernelSampler,
    TreeModel,
)
from shapiq.games.maskers._base import Masker

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


def test_the_coefficient_lock_survives_round_trips():
    estimate = ExactExplainer(additive_game(), SV()).estimate()
    for clone in (copy.deepcopy(estimate), pickle.loads(pickle.dumps(estimate))):
        with pytest.raises(ValueError, match="read-only"):
            clone._values[..., 0] = 7.0
        assert float(clone[(0,)]) == pytest.approx(float(estimate[(0,)]))


def test_tree_configuration_arrays_are_locked_and_owned():
    thresholds = np.array([0.5, 0.0, 0.0])
    tree = TreeModel(
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
        features=[0, -1, -1],
        thresholds=thresholds,
        values=[0.0, 1.0, 2.0],
    )
    with pytest.raises(ValueError, match="read-only"):
        tree.thresholds[0] = 9.0
    thresholds[0] = 9.0  # the caller's alias is severed
    assert float(tree.thresholds[0]) == 0.5
    game = InterventionalTreeGame(tree, inputs=np.array([0.2, 0.4]), baseline=np.array([0.9, 0.1]))
    with pytest.raises(ValueError, match="read-only"):
        game.leaf_constraints[0].values[0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        game.inputs[0] = 5.0


def test_custom_maskers_are_frozen_too():
    class _NullMasker(Masker):
        def __init__(self) -> None:
            self.n_players = 3
            self.target_shape = ()

        def _mask(self, coalitions):
            return coalitions.to_dense()

    masker = _NullMasker()
    with pytest.raises(AttributeError, match="frozen"):
        masker.n_players = 99
