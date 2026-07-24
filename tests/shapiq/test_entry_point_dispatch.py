"""Tests for the index-dispatched estimator entry points."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import (
    FBII,
    FSII,
    SII,
    SV,
    CallableGame,
    Estimate,
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


def order_one(source):
    read = source.__getitem__ if isinstance(source, Estimate) else source
    return jnp.stack([read((player,)) for player in range(N_PLAYERS)], axis=-1)


def test_unsupported_indices_are_rejected_with_teaching_errors():
    with pytest.raises(TypeError, match="PermutationSampling does not support 'FSII'"):
        PermutationSampling(cubic_game(), FSII(order=2))
    with pytest.raises(TypeError, match="Regression does not support 'SII'"):
        Regression(cubic_game(), SII(order=2))


def test_permutation_sampling_sv_equals_order_one_sii():
    shapley = PermutationSampling(cubic_game(), SV(), random_state=4).estimate(50)
    interactions = PermutationSampling(cubic_game(), SII(order=1), random_state=4).estimate(50)
    assert shapley.evidence == interactions.evidence
    assert jnp.allclose(
        order_one(shapley),
        order_one(interactions),
        atol=1e-6,
    )
    assert shapley.index == SV()
    assert interactions.index == SII(order=1)


def test_regression_with_sv_is_kernelshap():
    exact = order_one(ExactExplainer(cubic_game(), SV()).explain())
    approximator = Regression(cubic_game(), SV(), random_state=1)
    estimate = order_one(approximator.estimate(2 + 3000))
    assert jnp.allclose(estimate, exact, atol=0.05)


def test_regression_sv_matches_order_one_fsii_over_the_same_stream():
    budget = 2 + 24
    shapley = Regression(cubic_game(), SV(), random_state=3, deduplicate=True).estimate(budget)
    faithful = Regression(
        cubic_game(), FSII(order=1), random_state=3, deduplicate=True
    ).estimate(budget)
    assert shapley.evidence == faithful.evidence
    assert jnp.allclose(
        order_one(shapley),
        order_one(faithful),
        atol=1e-6,
    )
    assert shapley.index == SV()


def test_explanations_default_to_undirected_interactions():
    explainer = ExactExplainer(cubic_game(), SII(order=2))
    assert not hasattr(explainer, "orientation")  # orientation is no index concern
    assert not hasattr(SII(order=2), "orientation")
    explanation = explainer.explain()
    assert explanation.orientation == "undirected"
    assert jnp.allclose(explanation((2, 0)), explanation((0, 2)), atol=0)  # keys are sorted


def test_repr_names_the_entry_point_and_index():
    approximator = Regression(cubic_game(), FSII(order=2))
    text = repr(approximator)
    assert "Regression" in text
    assert "interaction_index='FSII'" in text


def test_subclasses_flow_to_their_parents_entry_points():
    class MySII(SII): ...

    class MyFSII(FSII): ...

    class MyFBII(FBII): ...

    # inheritance is a feature: a subclass inherits its parent's estimator
    # through the MRO and answers for its own semantics
    assert PermutationSampling(cubic_game(), MySII(order=2)).interaction_index == "SII"
    assert Regression(cubic_game(), MyFSII(order=2)).interaction_index == "FSII"
    subclassed = order_one(ExactExplainer(cubic_game(), MyFBII(order=2)).explain())
    reference = order_one(ExactExplainer(cubic_game(), FBII(order=2)).explain())
    assert jnp.allclose(subclassed, reference, atol=1e-6)


def test_explanations_carry_the_index_object():
    explanation = ExactExplainer(cubic_game(), SII(order=2)).explain()
    assert explanation.index == SII(order=2)
    assert explanation.interaction_index == "SII"
    assert "SII(order=2)" in repr(explanation)
