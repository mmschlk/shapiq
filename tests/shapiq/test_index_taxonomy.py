"""Tests for index-class taxonomy: order resolution, value preservation, open names."""

from __future__ import annotations

import copy

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
    PermutationSampling,
    Regression,
)

N_PLAYERS = 5

ALL_INDICES = [
    SV,
    BV,
    SII,
    BII,
    CHII,
    STII,
    KSII,
    FSII,
    FBII,
    KADDSHAP,
    Moebius,
    CoMoebius,
    SGV,
    BGV,
    CHGV,
    IGV,
    EGV,
    JointSV,
]


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


@pytest.mark.parametrize("index", ALL_INDICES)
def test_indices_are_singleton_values(index):
    with pytest.raises(TypeError, match="already the index"):
        index()
    with pytest.raises(TypeError, match="already the index"):
        index(order=2)
    assert type(index)() is index  # constructing the hidden class yields the singleton
    assert copy.deepcopy(index) is index


def test_values_fix_the_order_at_one():
    assert SV.resolve_order(None, n_players=N_PLAYERS) == 1
    assert BV.resolve_order(1, n_players=N_PLAYERS) == 1
    with pytest.raises(ValueError, match="fixed at 1"):
        SV.resolve_order(2, n_players=N_PLAYERS)


def test_interaction_indices_require_an_explicit_order():
    with pytest.raises(TypeError, match="pass order= to the explainer"):
        SII.resolve_order(None, n_players=N_PLAYERS)
    assert SII.resolve_order(3, n_players=N_PLAYERS) == 3
    with pytest.raises(ValueError, match="must not exceed"):
        STII.resolve_order(N_PLAYERS + 1, n_players=N_PLAYERS)
    with pytest.raises(ValueError, match="at least 1"):
        SII.resolve_order(0, n_players=N_PLAYERS)


def test_transforms_default_to_all_orders():
    assert Moebius.resolve_order(None, n_players=N_PLAYERS) == N_PLAYERS
    assert CoMoebius.resolve_order(2, n_players=N_PLAYERS) == 2


@pytest.mark.parametrize(
    "index_type",
    [SII, BII, CHII, KADDSHAP, SGV, BGV, CHGV],
)
def test_value_preserving_indices_keep_the_value_at_higher_orders(index_type):
    assert index_type.preserves_value
    game = random_table_game()
    restricted = order_one(ExactExplainer(game, index_type, order=2).explain())
    value = order_one(ExactExplainer(random_table_game(), index_type.generalizes).explain())
    assert jnp.allclose(restricted, value, atol=1e-4)


@pytest.mark.parametrize("index_type", [STII, KSII, FSII, FBII, JointSV])
def test_non_preserving_indices_leave_the_value_at_higher_orders(index_type):
    assert not index_type.preserves_value
    game = random_table_game()
    restricted = order_one(ExactExplainer(game, index_type, order=2).explain())
    value = order_one(ExactExplainer(random_table_game(), index_type.generalizes).explain())
    assert not jnp.allclose(restricted, value, atol=1e-3)


class _RenamedShapley:
    """A third-party cardinal index with its own name and Shapley weights."""

    name = "MySV"
    order_semantics = "coverage"
    includes_empty_interaction = False
    min_interaction_size = 1
    preserves_value = True
    generalizes = SV

    def resolve_order(self, order, *, n_players):
        return SV.resolve_order(order, n_players=n_players)

    def derivative_weights(self, n_players, interaction_size, *, order):
        return SV.derivative_weights(n_players, interaction_size, order=order)


def test_custom_named_indices_work_end_to_end():
    game = random_table_game()
    explanation = ExactExplainer(game, _RenamedShapley()).explain()
    reference = ExactExplainer(random_table_game(), SV).explain()
    assert explanation.interaction_index == "MySV"
    assert jnp.allclose(order_one(explanation), order_one(reference), atol=1e-6)


ORDER_FREE = [SV, BV, Moebius, CoMoebius]


@pytest.mark.parametrize("index_type", ALL_INDICES)
def test_empty_interaction_metadata_matches_explanations(index_type):
    order = None if index_type in ORDER_FREE else 2
    explanation = ExactExplainer(random_table_game(), index_type, order=order).explain()
    if index_type.includes_empty_interaction:
        explanation(())  # the order-0 attribution exists
    else:
        with pytest.raises(KeyError, match="not represented"):
            explanation(())


@pytest.mark.parametrize(
    "index_type",
    [SII, BII, CHII, SGV, BGV, CHGV, IGV, EGV, Moebius, CoMoebius],
)
def test_coverage_semantics_hold_numerically(index_type):
    assert index_type.order_semantics == "coverage"
    first = order_one(ExactExplainer(random_table_game(), index_type, order=1).explain())
    second = order_one(ExactExplainer(random_table_game(), index_type, order=2).explain())
    assert jnp.allclose(first, second, atol=1e-6)


def test_values_declare_their_singleton_marginal_weights():
    assert jnp.allclose(
        SV.marginal_weights(N_PLAYERS, 1, order=1),
        SGV.marginal_weights(N_PLAYERS, 1, order=2),
    )
    assert jnp.allclose(
        BV.marginal_weights(N_PLAYERS, 1, order=1),
        BGV.marginal_weights(N_PLAYERS, 1, order=2),
    )
    with pytest.raises(ValueError, match="single players only"):
        SV.marginal_weights(N_PLAYERS, 2, order=1)


def test_regression_kernels_declare_their_constraint_structure():
    assert jnp.allclose(FBII.regression_kernel(4), jnp.ones(5))
    shapley_kernel = FSII.regression_kernel(N_PLAYERS)
    assert shapley_kernel[0] == 0.0
    assert shapley_kernel[-1] == 0.0


def test_index_classes_print_as_bare_names():
    assert repr(SII) == "SII"
    assert repr(SV) == "SV"


def test_lookalike_indices_are_rejected_at_closed_entry_points():
    class MySII(type(SII)): ...

    class MyFSII(type(FSII)): ...

    class MyFBII(type(FBII)): ...

    game = random_table_game()
    with pytest.raises(TypeError, match="index singleton"):
        PermutationSampling(game, MySII(), order=2)
    with pytest.raises(TypeError, match="index singleton"):
        Regression(game, MyFSII(), order=2)
    with pytest.raises(TypeError, match="builds on FBII"):
        ExactExplainer(game, MyFBII(), order=2)
