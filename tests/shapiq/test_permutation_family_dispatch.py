"""Tests for the single-dispatched permutation walk families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import pytest

from shapiq import FBII, SII, STII, SV, CallableGame, PermutationSampling
from shapiq.explainers._permutation import (
    PermutationFamily,
    _chain_walk,
    _explain_shapley_values,
    permutation_family,
)
from shapiq.interactions import ExtensionalEquality

N_PLAYERS = 5
WEIGHTS = jnp.asarray([0.7, -1.3, 0.1, 2.0, -0.4])
PAIRS = jnp.asarray(
    [
        [0.0, 0.5, -1.0, 0.0, 0.3],
        [0.5, 0.0, 0.2, -0.7, 0.0],
        [-1.0, 0.2, 0.0, 0.4, 0.9],
        [0.0, -0.7, 0.4, 0.0, -0.2],
        [0.3, 0.0, 0.9, -0.2, 0.0],
    ],
)


def quadratic_game():
    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)

    return CallableGame(fn=fn, n_players=N_PLAYERS)


def order_one(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


def test_the_registry_carries_the_shipped_families_atomically():
    assert {SV, SII, STII} <= set(permutation_family.registry)
    for index in (SV(), SII(order=2), STII(order=2)):
        family = permutation_family(index)
        assert callable(family.walk)
        assert callable(family.explain)


def test_unregistered_indices_get_the_registry_derived_teaching_error():
    with pytest.raises(TypeError, match=r"does not support 'FBII'.*SII, STII, SV"):
        PermutationSampling(quadratic_game(), FBII(order=2))


def test_subclasses_inherit_their_parents_family_through_the_mro():
    class MySII(SII):
        pass

    # singledispatch resolves along the MRO: a subclass rides its parent's
    # complete family and answers for its own semantics
    assert permutation_family(MySII(order=2)) == permutation_family(SII(order=2))
    budget = 2 + 24 * 2 * (N_PLAYERS - 1)
    subclassed = PermutationSampling(
        quadratic_game(),
        MySII(order=2),
        random_state=3,
    ).sample(budget)
    reference = PermutationSampling(quadratic_game(), SII(order=2), random_state=3).sample(budget)
    assert subclassed.state == reference.state  # same walks, bit-identical
    assert jnp.allclose(
        order_one(subclassed.explain()),
        order_one(reference.explain()),
        atol=1e-6,
    )


@dataclass(frozen=True, eq=False)
class _MirroredSV(ExtensionalEquality):
    """A third-party index estimating Shapley values under its own name."""

    name: ClassVar[str] = "MirroredSV"
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[None] = None

    @property
    def order(self) -> int:
        return 1


@permutation_family.register
def _mirrored_family(index: _MirroredSV) -> PermutationFamily:
    del index
    return PermutationFamily(_chain_walk, _explain_shapley_values)


def test_registered_third_party_families_extend_the_method():
    budget = 2 + 12 * (N_PLAYERS - 1)
    mirrored = PermutationSampling(quadratic_game(), _MirroredSV(), random_state=3).sample(budget)
    reference = PermutationSampling(quadratic_game(), SV(), random_state=3).sample(budget)
    assert mirrored.state == reference.state  # same walks, bit-identical
    explanation = mirrored.explain()
    assert explanation.interaction_index == "MirroredSV"
    assert jnp.allclose(order_one(explanation), order_one(reference.explain()), atol=1e-6)
    # the teaching error now names the registered family too
    with pytest.raises(TypeError, match="_MirroredSV"):
        PermutationSampling(quadratic_game(), FBII(order=2))
