"""Tests for vector-valued games across the explainer families."""

from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import pytest

from shapiq import (
    FSII,
    SII,
    SV,
    CallableGame,
    DenseExplanationArray,
    Estimate,
    ExactExplainer,
    PermutationSampling,
    Regression,
)

N_PLAYERS = 5
KERNEL_SEEDS = 2
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


def quadratic_from_masks(masks):
    return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)


def cubic_from_masks(masks):
    return quadratic_from_masks(masks) + 1.5 * masks[..., 0] * masks[..., 1] * masks[..., 2]


def stacked_from_masks(masks):
    return jnp.stack([quadratic_from_masks(masks), cubic_from_masks(masks)], axis=-1)


def scalar_game(mask_fn):
    return CallableGame(
        fn=lambda c: mask_fn(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def vector_game():
    return CallableGame(
        fn=lambda c: stacked_from_masks(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
        value_shape=(2,),
    )


def order_one(source):
    read = source.__getitem__ if isinstance(source, Estimate) else source
    return jnp.stack([read((player,)) for player in range(N_PLAYERS)], axis=-2)


def test_exact_vector_values_match_per_component_scalar_runs():
    vector = ExactExplainer(vector_game(), SV()).estimate().view
    quadratic = ExactExplainer(scalar_game(quadratic_from_masks), SV()).estimate().view
    cubic = ExactExplainer(scalar_game(cubic_from_masks), SV()).estimate().view
    for player in range(N_PLAYERS):
        attribution = vector((player,))
        assert attribution.shape == (2,)
        assert jnp.allclose(attribution[0], quadratic((player,)), atol=1e-6)
        assert jnp.allclose(attribution[1], cubic((player,)), atol=1e-6)
    assert jnp.allclose(vector.baseline[0], quadratic.baseline, atol=1e-6)
    assert jnp.allclose(vector.baseline[1], cubic.baseline, atol=1e-6)


def test_exact_vector_interactions_match_per_component_scalar_runs():
    vector = ExactExplainer(vector_game(), SII(order=2)).estimate().view
    quadratic = ExactExplainer(scalar_game(quadratic_from_masks), SII(order=2)).estimate().view
    cubic = ExactExplainer(scalar_game(cubic_from_masks), SII(order=2)).estimate().view
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(vector(pair)[0], quadratic(pair), atol=1e-6)
        assert jnp.allclose(vector(pair)[1], cubic(pair), atol=1e-6)


def test_exact_vector_fsii_matches_per_component_scalar_runs():
    vector = ExactExplainer(vector_game(), FSII(order=2)).estimate().view
    quadratic = ExactExplainer(scalar_game(quadratic_from_masks), FSII(order=2)).estimate().view
    cubic = ExactExplainer(scalar_game(cubic_from_masks), FSII(order=2)).estimate().view
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(vector(pair)[0], quadratic(pair), atol=1e-5)
        assert jnp.allclose(vector(pair)[1], cubic(pair), atol=1e-5)


def test_sampled_vector_values_match_per_component_scalar_runs():
    budget = 2 + 20 * (N_PLAYERS - 1)
    vector = PermutationSampling(vector_game(), SV(), random_state=3).estimate(budget)
    quadratic = PermutationSampling(
        scalar_game(quadratic_from_masks), SV(), random_state=3
    ).estimate(budget)
    cubic = PermutationSampling(scalar_game(cubic_from_masks), SV(), random_state=3).estimate(
        budget,
    )
    for player in range(N_PLAYERS):
        assert jnp.allclose(vector[(player,)][0], quadratic[(player,)], atol=1e-6)
        assert jnp.allclose(vector[(player,)][1], cubic[(player,)], atol=1e-6)


def test_sampled_vector_values_are_efficient_per_component():
    grand = stacked_from_masks(jnp.ones(N_PLAYERS, dtype=jnp.float32))
    empty = stacked_from_masks(jnp.zeros(N_PLAYERS, dtype=jnp.float32))
    policy = PermutationSampling(vector_game(), SV(), random_state=0)
    estimate = policy.estimate(2 + 3 * (N_PLAYERS - 1))
    totals = jnp.sum(order_one(estimate), axis=-2)
    assert totals.shape == (2,)
    assert jnp.allclose(totals, grand - empty, atol=1e-5)
    assert jnp.allclose(estimate.view.baseline, empty, atol=1e-6)


def test_regression_fsii_vector_values_match_per_component_scalar_runs():
    budget = KERNEL_SEEDS + 24
    vector = Regression(
        vector_game(), FSII(order=2), random_state=0, deduplicate=True
    ).estimate(budget)
    quadratic = Regression(
        scalar_game(quadratic_from_masks), FSII(order=2), random_state=0, deduplicate=True
    ).estimate(budget)
    cubic = Regression(
        scalar_game(cubic_from_masks), FSII(order=2), random_state=0, deduplicate=True
    ).estimate(budget)
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(vector[pair][0], quadratic[pair], atol=1e-5)
        assert jnp.allclose(vector[pair][1], cubic[pair], atol=1e-5)
    for player in range(N_PLAYERS):
        assert jnp.allclose(vector[(player,)][0], quadratic[(player,)], atol=1e-5)


def test_partial_vector_walk_budgets_are_banked():
    def make():
        return PermutationSampling(vector_game(), SV(), random_state=5)

    complete = make().estimate(2 + 4 * (N_PLAYERS - 1))
    with_bank = make().estimate(2 + 4 * (N_PLAYERS - 1) + 1)
    assert with_bank.bank == 1
    assert with_bank.evidence.n_samples == complete.evidence.n_samples
    assert jnp.allclose(
        order_one(with_bank),
        order_one(complete),
        atol=1e-6,
    )


def test_vector_history_slices_evidence_states():
    policy = PermutationSampling(vector_game(), SV(), random_state=1)
    estimate = policy.refine(policy.estimate(2 + (N_PLAYERS - 1)), N_PLAYERS - 1)
    states = estimate.evidence.history()
    assert [state.n_samples for state in states] == [
        2 + (N_PLAYERS - 1),
        2 + 2 * (N_PLAYERS - 1),
    ]
    rolled = policy.at_evidence(estimate.evidence.rollback(1))
    # canonical state layout: value axes leading, sample axis last
    assert jnp.asarray(rolled.evidence.values).shape == (2, 2 + (N_PLAYERS - 1))


def test_misdeclared_value_shapes_are_rejected_at_the_boundary():
    scalar_declared_vector_output = CallableGame(
        fn=lambda c: stacked_from_masks(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )
    with pytest.raises(ValueError, match="declare value_shape"):
        PermutationSampling(scalar_declared_vector_output, SV(), random_state=0).estimate(16)
    vector_declared_scalar_output = CallableGame(
        fn=lambda c: quadratic_from_masks(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
        value_shape=(2,),
    )
    with pytest.raises(ValueError, match="value_shape"):
        ExactExplainer(vector_declared_scalar_output, SV()).estimate()


def test_explanations_validate_attribution_block_shapes():
    with pytest.raises(ValueError, match="expected"):
        DenseExplanationArray(
            attributions_by_order={1: jnp.zeros((N_PLAYERS, 3))},
            n_players=N_PLAYERS,
            index=SV(),
            order=1,
            value_shape=(2,),
        )
