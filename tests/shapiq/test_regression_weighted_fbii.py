"""Tests for the faithful weighted Banzhaf interaction index and its regression."""

from __future__ import annotations

from itertools import combinations, product

import jax.numpy as jnp
import pytest

from shapiq import (
    FBII,
    CallableGame,
    Estimate,
    ExactExplainer,
    PairedSampler,
    Regression,
    WeightedBII,
    WeightedBV,
    WeightedFBII,
)
from shapiq.sampling import ProductKernelSampler

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


def order_one(source):
    read = source.__getitem__ if isinstance(source, Estimate) else source
    return jnp.stack([read((player,)) for player in range(N_PLAYERS)], axis=-1)


def brute_force_weighted_fit(mask_fn, p, order):
    """Solve the product-measure weighted least squares fit on the full powerset."""
    masks = jnp.asarray(list(product([0.0, 1.0], repeat=N_PLAYERS)), dtype=jnp.float32)
    values = mask_fn(masks)
    centered = values - values[0]
    interactions = [
        combo for size in range(1, order + 1) for combo in combinations(range(N_PLAYERS), size)
    ]
    columns = [jnp.ones(masks.shape[0])]
    columns += [jnp.prod(masks[:, jnp.asarray(combo)], axis=-1) for combo in interactions]
    design = jnp.stack(columns, axis=-1)
    sizes = jnp.sum(masks, axis=-1)
    weights = p**sizes * (1.0 - p) ** (N_PLAYERS - sizes)
    sqrt_weights = jnp.sqrt(weights / jnp.max(weights))
    solution, *_ = jnp.linalg.lstsq(sqrt_weights[:, None] * design, sqrt_weights * centered)
    return {(): solution[0]} | dict(zip(interactions, solution[1:], strict=True))


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_exact_weighted_fbii_solves_the_product_measure_fit(p):
    expected = brute_force_weighted_fit(cubic_from_masks, p, order=2)
    explanation = ExactExplainer(game_from(cubic_from_masks), WeightedFBII(p=p, order=2)).explain()
    for interaction, coefficient in expected.items():
        assert jnp.allclose(explanation(interaction), coefficient, atol=1e-4)


def test_uniform_weighting_is_the_faithful_banzhaf_index():
    weighted = ExactExplainer(game_from(cubic_from_masks), WeightedFBII(p=0.5, order=2)).explain()
    banzhaf = ExactExplainer(game_from(cubic_from_masks), FBII(order=2)).explain()
    assert jnp.allclose(weighted(()), banzhaf(()), atol=1e-6)
    for size in (1, 2):
        for combo in combinations(range(N_PLAYERS), size):
            assert jnp.allclose(weighted(combo), banzhaf(combo), atol=1e-6)


def test_exact_order_one_is_the_weighted_banzhaf_value():
    fitted = ExactExplainer(game_from(cubic_from_masks), WeightedFBII(p=0.3, order=1)).explain()
    value = ExactExplainer(game_from(cubic_from_masks), WeightedBV(p=0.3)).explain()
    assert jnp.allclose(order_one(fitted), order_one(value), atol=1e-5)


@pytest.mark.filterwarnings("ignore::shapiq.errors.SamplingStallWarning")
def test_recovers_quadratic_games_exactly_once_identified():
    approximator = Regression(
        game_from(quadratic_from_masks),
        WeightedFBII(p=0.3, order=2),
        random_state=0,
        deduplicate=True,
    )
    estimate = approximator.estimate(SEEDS + 24)
    assert jnp.allclose(order_one(estimate), WEIGHTS, atol=1e-3)
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(estimate[(left, right)], PAIRS[left, right], atol=1e-3)
    # a 2-additive game's centered fit needs no intercept
    assert jnp.allclose(estimate[()], 0.0, atol=1e-3)


def test_converges_to_the_exact_faithful_weighted_interactions():
    exact = ExactExplainer(game_from(cubic_from_masks), WeightedFBII(p=0.7, order=2)).explain()
    approximator = Regression(
        game_from(cubic_from_masks),
        WeightedFBII(p=0.7, order=2),
        random_state=2,
    )
    estimate = approximator.estimate(SEEDS + 8000)
    for player in range(N_PLAYERS):
        assert jnp.allclose(estimate[(player,)], exact((player,)), atol=0.1)
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(estimate[pair], exact(pair), atol=0.1)


def test_pairing_follows_the_kernel_symmetry():
    game = game_from(cubic_from_masks)
    asymmetric = Regression(game, WeightedFBII(p=0.3, order=2))
    assert isinstance(asymmetric.sampler, ProductKernelSampler)
    assert asymmetric.unit_rows == 1
    uniform = Regression(game, WeightedFBII(p=0.5, order=2))
    assert isinstance(uniform.sampler, PairedSampler)
    explicit = Regression(game, WeightedFBII(p=0.3, order=2), paired=False)
    assert explicit.unit_rows == 1
    with pytest.raises(ValueError, match="not complement-symmetric"):
        Regression(game, WeightedFBII(p=0.3, order=2), paired=True)


def test_uniform_weighted_sampling_matches_the_fbii_stream():
    game = game_from(cubic_from_masks)
    weighted = Regression(
        game, WeightedFBII(p=0.5, order=2), random_state=6, deduplicate=True
    ).estimate(SEEDS + 28)
    banzhaf = Regression(game, FBII(order=2), random_state=6, deduplicate=True).estimate(
        SEEDS + 28,
    )
    assert weighted.evidence == banzhaf.evidence
    assert jnp.allclose(
        order_one(weighted),
        order_one(banzhaf),
        atol=1e-6,
    )


def test_weighted_fbii_equality_and_identity_chains():
    assert WeightedFBII(p=0.5, order=2) == FBII(order=2)
    assert WeightedFBII(p=0.3, order=1) == WeightedBV(p=0.3)
    assert WeightedFBII(p=0.3, order=1) == WeightedBII(p=0.3, order=1)
    assert WeightedFBII(p=0.3, order=2) != FBII(order=2)
    assert WeightedFBII(p=0.3, order=2) != WeightedFBII(p=0.4, order=2)
    assert WeightedFBII(p=0.3, order=2) != WeightedFBII(p=0.3, order=1)
    assert WeightedFBII(p=0.3, order=2) != WeightedBII(p=0.3, order=2)
    assert hash(WeightedFBII(p=0.5, order=2)) == hash(FBII(order=2))
    assert WeightedFBII(p=0.3, order=2).generalizes == WeightedBV(p=0.3)


@pytest.mark.filterwarnings("ignore::shapiq.errors.SamplingStallWarning")
def test_metadata_names_the_index():
    approximator = Regression(
        game_from(cubic_from_masks),
        WeightedFBII(p=0.3, order=2),
        random_state=0,
        deduplicate=True,
    )
    estimate = approximator.estimate(SEEDS + 24)
    assert estimate.index == WeightedFBII(p=0.3, order=2)
    assert estimate.view.order == 2
    assert WeightedFBII(p=0.3, order=2).includes_empty_interaction


def test_probability_validation_teaches_the_open_interval():
    with pytest.raises(ValueError, match="0 < p < 1"):
        WeightedFBII(p=0.0, order=2)
    with pytest.raises(TypeError, match="p must be a float"):
        WeightedFBII(p="half", order=2)
