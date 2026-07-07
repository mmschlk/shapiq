"""Tests for the index-dispatched estimator entry points."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import (
    FSII,
    SII,
    STII,
    SV,
    CallableGame,
    ExactExplainer,
    PermutationSampling,
    Regression,
)

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


def cubic_from_masks(masks):
    quadratic = masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)
    return quadratic + 1.5 * masks[..., 0] * masks[..., 1] * masks[..., 2]


def cubic_game():
    return CallableGame(
        fn=lambda c: cubic_from_masks(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def order_one(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


def test_unsupported_indices_are_rejected_with_teaching_errors():
    with pytest.raises(TypeError, match="PermutationSampling does not support 'FSII'"):
        PermutationSampling(cubic_game(), FSII(order=2))
    with pytest.raises(TypeError, match="Regression does not support 'SII'"):
        Regression(cubic_game(), SII(order=2))


def test_permutation_sampling_sv_equals_order_one_sii():
    shapley = PermutationSampling(cubic_game(), SV(), random_state=4).sample(50)
    interactions = PermutationSampling(cubic_game(), SII(order=1), random_state=4).sample(50)
    assert shapley.state == interactions.state
    assert jnp.allclose(
        order_one(shapley.explain()),
        order_one(interactions.explain()),
        atol=1e-6,
    )
    assert shapley.explain().interaction_index == "SV"
    assert interactions.explain().interaction_index == "SII"


def test_regression_with_sv_is_kernelshap():
    exact = order_one(ExactExplainer(cubic_game(), SV()).explain())
    approximator = Regression(cubic_game(), SV(), random_state=1)
    estimate = order_one(approximator.sample(2 + 3000).explain())
    assert jnp.allclose(estimate, exact, atol=0.05)


def test_regression_sv_matches_order_one_fsii_over_the_same_stream():
    budget = 2 + 24
    shapley = Regression(cubic_game(), SV(), random_state=3, deduplicate=True).sample(budget)
    faithful = Regression(cubic_game(), FSII(order=1), random_state=3, deduplicate=True).sample(
        budget,
    )
    assert shapley.state == faithful.state
    assert jnp.allclose(
        order_one(shapley.explain()),
        order_one(faithful.explain()),
        atol=1e-6,
    )
    assert shapley.explain().interaction_index == "SV"


def test_orientation_is_carried_by_the_index():
    assert SII(order=2).orientation == "undirected"
    assert STII(order=3).orientation == "undirected"
    approximator = PermutationSampling(cubic_game(), SV())
    assert approximator.orientation == "undirected"
    assert approximator.explain is not None  # explainers derive, never store, orientation
    explainer = ExactExplainer(cubic_game(), SV())
    assert explainer.orientation == "undirected"


def test_repr_names_the_entry_point_and_index():
    approximator = Regression(cubic_game(), FSII(order=2))
    text = repr(approximator)
    assert "Regression" in text
    assert "interaction_index='FSII'" in text
