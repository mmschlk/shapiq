"""Tests for the weighted Banzhaf value and interaction index (Marichal-Mathonet)."""

from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import BII, BV, CallableGame, ExactExplainer, WeightedBII, WeightedBV

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


def quadratic_from_masks(masks):
    return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)


def game_from(mask_fn):
    return CallableGame(
        fn=lambda c: mask_fn(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def random_table_game():
    rng = np.random.default_rng(7)
    table = rng.normal(size=2**N_PLAYERS)

    def mask_fn(masks):
        indices = jnp.asarray(masks @ (2.0 ** jnp.arange(N_PLAYERS)), dtype=jnp.int32)
        return jnp.asarray(table, dtype=jnp.float32)[indices]

    return mask_fn


def subset_mask(subset):
    mask = np.zeros(N_PLAYERS, dtype=np.float64)
    mask[list(subset)] = 1.0
    return jnp.asarray(mask, dtype=jnp.float32)


def discrete_derivative(mask_fn, interaction, base):
    total = 0.0
    for length in range(len(interaction) + 1):
        for part in combinations(interaction, length):
            sign = (-1) ** (len(interaction) - length)
            total += sign * float(mask_fn(subset_mask((*base, *part))))
    return total


def brute_force_weighted_banzhaf(mask_fn, interaction, p):
    """Sum binomially weighted discrete derivatives per Marichal-Mathonet eq. (15)."""
    others = [player for player in range(N_PLAYERS) if player not in interaction]
    free = len(others)
    total = 0.0
    for size in range(free + 1):
        for base in combinations(others, size):
            weight = p**size * (1.0 - p) ** (free - size)
            total += weight * discrete_derivative(mask_fn, interaction, base)
    return total


def order_one(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_exact_weighted_banzhaf_interactions_match_brute_force(p):
    mask_fn = random_table_game()
    explanation = ExactExplainer(game_from(mask_fn), WeightedBII(p=p, order=2)).estimate().view
    for player in (0, 3):
        expected = brute_force_weighted_banzhaf(mask_fn, (player,), p)
        assert jnp.allclose(explanation((player,)), expected, atol=1e-4)
    for pair in [(0, 1), (2, 4)]:
        expected = brute_force_weighted_banzhaf(mask_fn, pair, p)
        assert jnp.allclose(explanation(pair), expected, atol=1e-4)


def test_weighted_banzhaf_recovers_the_moebius_basis_of_quadratic_games():
    # binomial masses sum to one, so additive parts and pair masses are exact at any p
    explanation = ExactExplainer(game_from(quadratic_from_masks), WeightedBII(p=0.3, order=2))
    result = explanation.estimate().view
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(result((left, right)), PAIRS[left, right], atol=1e-4)


def test_uniform_weighting_is_the_banzhaf_index():
    mask_fn = random_table_game()
    weighted = ExactExplainer(game_from(mask_fn), WeightedBII(p=0.5, order=2)).estimate().view
    banzhaf = ExactExplainer(game_from(mask_fn), BII(order=2)).estimate().view
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(weighted(pair), banzhaf(pair), atol=1e-6)
    assert jnp.allclose(order_one(weighted), order_one(banzhaf), atol=1e-6)


def test_order_one_weighted_interactions_are_the_weighted_value():
    mask_fn = random_table_game()
    restricted = ExactExplainer(game_from(mask_fn), WeightedBII(p=0.3, order=1)).estimate().view
    value = ExactExplainer(game_from(mask_fn), WeightedBV(p=0.3)).estimate().view
    assert jnp.allclose(order_one(restricted), order_one(value), atol=1e-6)
    assert WeightedBII(p=0.3, order=1).generalizes == WeightedBV(p=0.3)


def test_weighted_instances_carry_their_parameter_into_equality():
    assert WeightedBV(p=0.5) == BV()
    assert WeightedBII(p=0.5, order=2) == BII(order=2)
    assert WeightedBII(p=0.3, order=1) == WeightedBV(p=0.3)
    assert WeightedBV(p=0.3) != BV()
    assert WeightedBV(p=0.3) != WeightedBV(p=0.4)
    assert WeightedBII(p=0.3, order=2) != WeightedBII(p=0.3, order=1)
    assert len({BV(), WeightedBV(p=0.5), WeightedBII(p=0.5, order=1)}) == 1
    assert hash(WeightedBII(p=0.3, order=1)) == hash(WeightedBV(p=0.3))


def test_weighted_explanations_carry_the_index_name():
    explanation = ExactExplainer(game_from(quadratic_from_masks), WeightedBV(p=0.3)).estimate().view
    assert explanation.interaction_index == "WeightedBV"
    assert explanation.order == 1


def test_probability_validation_teaches_the_open_interval():
    for bad in (0.0, 1.0, -0.2, 1.7):
        with pytest.raises(ValueError, match="0 < p < 1"):
            WeightedBV(p=bad)
    with pytest.raises(ValueError, match="Moebius"):
        WeightedBII(p=0.0, order=2)
    with pytest.raises(TypeError, match="p must be a float"):
        WeightedBV(p="half")
