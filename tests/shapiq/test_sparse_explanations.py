"""Tests for sparse explanation arrays: index objects, value shapes, slicing."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import SII, SV, CoMoebius, DenseExplanationArray, SparseExplanationArray

N_PLAYERS = 3


def sii_sparse(*, with_default=True):
    return SparseExplanationArray(
        {(0,): 0.5, (1,): -1.0, (1, 2): -0.25},
        n_players=N_PLAYERS,
        index=SII(order=2),
        order=2,
        default_attribution=(lambda interaction: jnp.zeros(())) if with_default else None,
    )


def test_sparse_arrays_carry_the_index_object():
    explanation = SparseExplanationArray(
        {(0,): 0.5, (1, 2): -0.25},
        n_players=N_PLAYERS,
        index=SII(order=2),
        order=2,
    )
    assert explanation.index == SII(order=2)
    assert explanation.interaction_index == "SII"
    assert explanation((1, 2)) == -0.25
    assert "SII(order=2)" in repr(explanation)


def test_sparse_arrays_reject_name_strings_with_a_teaching_error():
    with pytest.raises(TypeError, match=r"pass index=shapiq\.SV"):
        SparseExplanationArray({(0,): 0.5}, n_players=N_PLAYERS, index="SV", order=1)


def test_index_and_explanation_orders_must_agree():
    with pytest.raises(ValueError, match="explanation records order 3"):
        SparseExplanationArray({(0,): 0.5}, n_players=N_PLAYERS, index=SII(order=2), order=3)


def test_sparse_value_shapes_are_validated():
    explanation = SparseExplanationArray(
        {(0,): jnp.asarray([1.0, 2.0])},
        n_players=N_PLAYERS,
        index=SV(),
        order=1,
        value_shape=(2,),
        baseline=jnp.asarray([0.5, 0.5]),
    )
    assert jnp.allclose(explanation((0,)), jnp.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="targets, then value_shape"):
        SparseExplanationArray(
            {(0,): jnp.asarray([1.0, 2.0, 3.0])},
            n_players=N_PLAYERS,
            index=SV(),
            order=1,
            value_shape=(2,),
        )
    with pytest.raises(ValueError, match="baseline has shape"):
        SparseExplanationArray(
            {(0,): jnp.asarray([1.0, 2.0])},
            n_players=N_PLAYERS,
            index=SV(),
            order=1,
            value_shape=(2,),
            baseline=jnp.asarray([0.5]),
        )


def test_batched_lookups_consult_stored_attributions():
    explanation = sii_sparse()
    singles = explanation(jnp.asarray([[0], [1], [2]]))
    assert singles.shape == (3,)
    assert jnp.allclose(singles, jnp.asarray([0.5, -1.0, 0.0]))  # (2,) unstored -> default
    pairs = explanation(jnp.asarray([[2, 1], [0, 1]]))  # rows normalize like tuples
    assert jnp.allclose(pairs, jnp.asarray([-0.25, 0.0]))
    grid = explanation(jnp.asarray([[[0], [1]], [[2], [0]]]))
    assert grid.shape == (2, 2)
    assert jnp.allclose(grid, jnp.asarray([[0.5, -1.0], [0.0, 0.5]]))


def test_batched_lookups_keep_the_dense_value_layout():
    explanation = SparseExplanationArray(
        {(0,): jnp.asarray([1.0, 2.0]), (2,): jnp.asarray([-1.0, 3.0])},
        n_players=N_PLAYERS,
        index=SV(),
        order=1,
        value_shape=(2,),
        default_attribution=lambda interaction: jnp.zeros(2),
    )
    batch = explanation(jnp.asarray([[0], [1], [2]]))
    assert batch.shape == (3, 2)
    assert jnp.allclose(batch, jnp.asarray([[1.0, 2.0], [0.0, 0.0], [-1.0, 3.0]]))


def test_missing_rows_without_a_default_raise_pointing_to_has():
    explanation = sii_sparse(with_default=False)
    assert jnp.allclose(explanation(jnp.asarray([[0], [1]])), jnp.asarray([0.5, -1.0]))
    with pytest.raises(KeyError, match="probe availability with has"):
        explanation(jnp.asarray([[0], [2]]))
    with pytest.raises(KeyError, match="probe availability with has"):
        explanation((2,))
    availability = explanation.has(jnp.asarray([[0], [2]]))
    assert availability.tolist() == [True, False]


def test_the_represented_window_comes_from_the_index():
    explanation = sii_sparse()
    with pytest.raises(KeyError, match="defines no order-0 attribution"):
        explanation(())
    with pytest.raises(KeyError, match="sizes 1 to 2, not 3"):
        explanation((0, 1, 2))
    assert not bool(explanation.has(()))
    assert not bool(explanation.has((0, 1, 2)))
    # an index that attributes to the empty interaction keeps its order 0
    comoebius = SparseExplanationArray(
        {(): 1.5, (0,): 0.5},
        n_players=N_PLAYERS,
        index=CoMoebius(order=1),
        order=1,
    )
    assert comoebius(()) == 1.5
    assert bool(comoebius.has(()))


def test_construction_rejects_attributions_outside_the_window():
    with pytest.raises(KeyError, match="defines no order-0 attribution"):
        SparseExplanationArray({(): 1.0}, n_players=N_PLAYERS, index=SV(), order=1)


def test_iteration_yields_only_representable_interactions():
    explanation = sii_sparse()
    interactions = list(explanation.iter_interactions())
    assert interactions[0] == (0,)  # starts at the index's smallest represented size
    for interaction in interactions:
        explanation(interaction)  # every yielded interaction can be looked up
    comoebius = SparseExplanationArray(
        {(): 1.5},
        n_players=N_PLAYERS,
        index=CoMoebius(order=1),
        order=1,
        default_attribution=lambda interaction: jnp.zeros(()),
    )
    assert next(iter(comoebius.iter_interactions())) == ()
    # an explicit lower bound still wins over the index default
    assert next(iter(explanation.iter_interactions(min_order=0))) == ()


def test_array_lookup_argument_misuse_teaches():
    explanation = sii_sparse()
    with pytest.raises(TypeError, match="final interaction-members axis"):
        explanation(5)
    with pytest.raises(TypeError, match="integer dtype"):
        explanation(jnp.asarray([[0.5]]))


def test_dense_window_messages_distinguish_index_from_storage():
    dense = DenseExplanationArray(
        {2: jnp.zeros(3)},  # only the pairs block is stored
        n_players=N_PLAYERS,
        index=SII(order=2),
        order=2,
    )
    with pytest.raises(KeyError, match="defines no order-0 attribution"):
        dense(())
    with pytest.raises(KeyError, match="no order-1 attributions are stored"):
        dense((0,))
    assert not bool(dense.has((0,)))


def test_sparse_indexing_slices_attributions_and_baseline():
    explanation = SparseExplanationArray(
        {(0,): jnp.asarray([1.0, 2.0]), (2,): jnp.asarray([-1.0, 3.0])},
        n_players=N_PLAYERS,
        index=SV(),
        order=1,
        shape=(2,),
        baseline=jnp.asarray([0.25, 0.75]),
    )
    first = explanation[0]
    assert first.shape == ()
    assert jnp.allclose(first((0,)), 1.0)
    assert jnp.allclose(first((2,)), -1.0)
    assert jnp.allclose(first.baseline, 0.25)
