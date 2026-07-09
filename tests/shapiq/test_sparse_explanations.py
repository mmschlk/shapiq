"""Tests for sparse explanation arrays: index objects, value shapes, slicing."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import SII, SV, SparseExplanationArray

N_PLAYERS = 3


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
