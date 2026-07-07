"""Tests for the kernel-parametrized regression family."""

from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import pytest

from shapiq import (
    FBII,
    KADDSHAP,
    CallableGame,
    ExactExplainer,
    InsufficientSamplesError,
    Regression,
    define_regression_index,
)

N_PLAYERS = 5
SEEDS = 2
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


def game_from(mask_fn):
    return CallableGame(
        fn=lambda c: mask_fn(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def all_interactions(order):
    for size in range(1, order + 1):
        yield from combinations(range(N_PLAYERS), size)


def center_kernel(n_players):
    sizes = jnp.arange(n_players + 1, dtype=jnp.float32)
    return sizes * (n_players - sizes)


def test_sampled_kadd_shap_recovers_quadratic_games_once_identified():
    game = game_from(quadratic_from_masks)
    exact = ExactExplainer(game, KADDSHAP(order=2)).explain()
    approximator = Regression(game, KADDSHAP(order=2), random_state=0, deduplicate=True)
    explanation = approximator.sample(SEEDS + 24).explain()
    assert explanation.interaction_index == "kADD-SHAP"
    for interaction in all_interactions(2):
        assert jnp.allclose(explanation(interaction), exact(interaction), atol=1e-3)


def test_sampled_kadd_shap_converges_on_non_additive_games():
    game = game_from(cubic_from_masks)
    exact = ExactExplainer(game, KADDSHAP(order=2)).explain()
    explanation = Regression(game, KADDSHAP(order=2), random_state=2).sample(SEEDS + 3000).explain()
    for interaction in all_interactions(2):
        assert jnp.allclose(explanation(interaction), exact(interaction), atol=0.25)


def test_sampled_kadd_shap_gates_on_identification():
    approximator = Regression(
        game_from(cubic_from_masks),
        KADDSHAP(order=2),
        random_state=0,
        deduplicate=True,
    )
    with pytest.raises(InsufficientSamplesError, match="not yet identified"):
        approximator.sample(SEEDS + 6).explain()


def test_defined_regression_index_recovers_quadratic_games_once_identified():
    index = define_regression_index("CenterFit", kernel=center_kernel, order=2)
    game = game_from(quadratic_from_masks)
    exact = ExactExplainer(game, index).explain()
    approximator = Regression(game, index, random_state=1, deduplicate=True)
    explanation = approximator.sample(SEEDS + 24).explain()
    assert explanation.interaction_index == "CenterFit"
    assert jnp.allclose(explanation(()), exact(()), atol=1e-5)
    for interaction in all_interactions(2):
        assert jnp.allclose(explanation(interaction), exact(interaction), atol=1e-3)


def test_paired_sampling_requires_a_symmetric_kernel():
    def lopsided_kernel(n_players):
        kernel = jnp.zeros(n_players + 1)
        return kernel.at[1].set(3.0).at[2].set(1.0).at[n_players - 1].set(0.5)

    index = define_regression_index("Lopsided", kernel=lopsided_kernel, order=1)
    game = game_from(quadratic_from_masks)
    with pytest.raises(ValueError, match="paired=False"):
        Regression(game, index, random_state=0)
    approximator = Regression(game, index, random_state=0, paired=False)
    approximator.sample(SEEDS + 12).explain()


def test_declared_kernels_must_zero_the_constrained_endpoints():
    index = define_regression_index(
        "EndpointHeavy",
        kernel=lambda n_players: jnp.ones(n_players + 1),
        order=1,
    )
    with pytest.raises(ValueError, match="must be zero"):
        Regression(game_from(quadratic_from_masks), index)


def test_regression_rejects_indices_without_a_kernel():
    with pytest.raises(TypeError, match="Regression does not support 'FBII'"):
        Regression(game_from(quadratic_from_masks), FBII(order=2))
