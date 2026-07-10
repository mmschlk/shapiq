"""Tests for extensional index equality, value preservation, and open names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    BGV,
    BII,
    BV,
    CHGV,
    CHII,
    EGV,
    FBII,
    FSII,
    IGV,
    KADDSHAP,
    KSII,
    SGV,
    SII,
    STII,
    SV,
    CallableGame,
    CoMoebius,
    ExactExplainer,
    JointSV,
    Moebius,
    WeightedBII,
    WeightedBV,
    WeightedFBII,
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


def random_table_game():
    rng = np.random.default_rng(42)
    table = rng.normal(size=2**N_PLAYERS)

    def mask_fn(masks):
        indices = jnp.asarray(masks @ (2.0 ** jnp.arange(N_PLAYERS)), dtype=jnp.int32)
        return jnp.asarray(table, dtype=jnp.float32)[indices]

    return CallableGame(
        fn=lambda c: mask_fn(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def order_one(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


SHAPLEY_ORDER_ONE = [SII, CHII, STII, KSII, FSII, KADDSHAP, SGV, CHGV, JointSV]
BANZHAF_ORDER_ONE = [BII, FBII, BGV]


def test_order_one_instances_equal_the_shapley_value():
    for index_type in SHAPLEY_ORDER_ONE:
        instance = index_type(order=1)
        assert instance == SV()
        assert SV() == instance
        assert hash(instance) == hash(SV())
    assert len({SV(), *(index_type(order=1) for index_type in SHAPLEY_ORDER_ONE)}) == 1


def test_order_one_instances_equal_the_banzhaf_value():
    for index_type in BANZHAF_ORDER_ONE:
        assert index_type(order=1) == BV()
        assert hash(index_type(order=1)) == hash(BV())
    assert SV() != BV()


def test_higher_orders_and_transforms_stay_distinct():
    assert SII(order=2) != SV()
    assert SII(order=2) != SII(order=1)
    assert SII(order=2) != CHII(order=2)  # different operators at order 2
    assert SII(order=2) == SII(order=2)
    assert Moebius(order=1) != SV()  # the Moebius transform generalizes nothing
    assert IGV(order=1) != SV()
    assert EGV(order=1) != CoMoebius(order=1)  # extensional only order-1-vs-value


@pytest.mark.parametrize(
    "index_type",
    [SII, BII, CHII, KADDSHAP, SGV, BGV, CHGV],
)
def test_value_preserving_indices_keep_the_value_at_higher_orders(index_type):
    index = index_type(order=2)
    assert index.preserves_value
    game = random_table_game()
    restricted = order_one(ExactExplainer(game, index).explain())
    value = order_one(ExactExplainer(random_table_game(), index.generalizes).explain())
    assert jnp.allclose(restricted, value, atol=1e-4)


@pytest.mark.parametrize("index_type", [STII, KSII, FSII, FBII, JointSV])
def test_non_preserving_indices_leave_the_value_at_higher_orders(index_type):
    index = index_type(order=2)
    assert not index.preserves_value
    game = random_table_game()
    restricted = order_one(ExactExplainer(game, index).explain())
    value = order_one(ExactExplainer(random_table_game(), index.generalizes).explain())
    assert not jnp.allclose(restricted, value, atol=1e-3)


@dataclass(frozen=True, eq=False)
class _RenamedShapley(ExtensionalEquality):
    """A third-party cardinal index with its own name and Shapley weights."""

    order: int = 1

    name: ClassVar = "MySV"
    order_semantics: ClassVar = "coverage"
    orientation: ClassVar = "undirected"
    min_interaction_size: ClassVar = 1
    preserves_value: ClassVar = True
    generalizes: ClassVar = SV()

    def derivative_weights(self, n_players: int, interaction_size: int):
        return SV().derivative_weights(n_players, interaction_size)


def test_custom_named_indices_work_end_to_end():
    game = random_table_game()
    explanation = ExactExplainer(game, _RenamedShapley()).explain()
    reference = ExactExplainer(random_table_game(), SV()).explain()
    assert explanation.interaction_index == "MySV"
    assert jnp.allclose(order_one(explanation), order_one(reference), atol=1e-6)
    # opting into the equality mixin makes the order-1 identity hold for it too
    assert _RenamedShapley() == SV()


ALL_INSTANCES = [
    SV(),
    BV(),
    WeightedBV(p=0.3),
    SII(order=2),
    BII(order=2),
    WeightedBII(p=0.3, order=2),
    CHII(order=2),
    STII(order=2),
    KSII(order=2),
    FSII(order=2),
    FBII(order=2),
    WeightedFBII(p=0.3, order=2),
    KADDSHAP(order=2),
    Moebius(),
    CoMoebius(),
    SGV(order=2),
    BGV(order=2),
    CHGV(order=2),
    IGV(order=2),
    EGV(order=2),
    JointSV(order=2),
]


@pytest.mark.parametrize("index", ALL_INSTANCES, ids=repr)
def test_empty_interaction_metadata_matches_explanations(index):
    # includes_empty_interaction is derived, never stored separately
    assert index.includes_empty_interaction == (index.min_interaction_size == 0)
    explanation = ExactExplainer(random_table_game(), index).explain()
    if index.includes_empty_interaction:
        explanation(())  # the order-0 attribution exists
    else:
        with pytest.raises(KeyError, match="defines no order-0 attribution"):
            explanation(())


@pytest.mark.parametrize(
    "index_type",
    [SII, BII, WeightedBII, CHII, SGV, BGV, CHGV, IGV, EGV, Moebius, CoMoebius],
)
def test_coverage_semantics_hold_numerically(index_type):
    assert index_type(order=1).order_semantics == "coverage"
    first = order_one(ExactExplainer(random_table_game(), index_type(order=1)).explain())
    second = order_one(ExactExplainer(random_table_game(), index_type(order=2)).explain())
    assert jnp.allclose(first, second, atol=1e-6)
