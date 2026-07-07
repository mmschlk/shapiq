"""Tests for the exact explainer across all supported indices."""

from __future__ import annotations

from itertools import combinations
from math import comb

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import BII, BV, FSII, SII, STII, SV, CallableGame, ExactExplainer, PermutationSampling

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


def cubic_from_masks(masks):
    return quadratic_from_masks(masks) + 1.5 * masks[..., 0] * masks[..., 1] * masks[..., 2]


def game_from(mask_fn):
    return CallableGame(
        fn=lambda c: mask_fn(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


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


def brute_force_base_interaction(mask_fn, interaction, weight_of_t):
    others = [p for p in range(N_PLAYERS) if p not in interaction]
    total = 0.0
    for size in range(len(others) + 1):
        for base in combinations(others, size):
            total += weight_of_t(size) * discrete_derivative(mask_fn, interaction, base)
    return total


def sii_weight(interaction_size):
    def weight(t):
        free = N_PLAYERS - interaction_size
        return 1.0 / ((free + 1) * comb(free, t))

    return weight


def bii_weight(interaction_size):
    def weight(t):
        return 2.0 ** -(N_PLAYERS - interaction_size)

    return weight


def stii_top_weight(order):
    def weight(t):
        return (order / N_PLAYERS) / comb(N_PLAYERS - 1, t)

    return weight


def random_table_game():
    rng = np.random.default_rng(42)
    table = rng.normal(size=2**N_PLAYERS)

    def mask_fn(masks):
        indices = jnp.asarray(masks @ (2.0 ** jnp.arange(N_PLAYERS)), dtype=jnp.int32)
        return jnp.asarray(table, dtype=jnp.float32)[indices]

    return mask_fn


def order_one(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


def test_exact_shapley_values_match_closed_form():
    explanation = ExactExplainer(game_from(quadratic_from_masks), SV()).explain()
    assert jnp.allclose(order_one(explanation), WEIGHTS + 0.5 * jnp.sum(PAIRS, axis=1), atol=1e-5)


def test_exact_shapley_and_banzhaf_differ_on_cubic_games():
    game = game_from(cubic_from_masks)
    shapley = order_one(ExactExplainer(game, SV()).explain())
    banzhaf = order_one(ExactExplainer(game_from(cubic_from_masks), BV()).explain())
    base = WEIGHTS + 0.5 * jnp.sum(PAIRS, axis=1)
    cubic_members = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(shapley, base + 1.5 / 3 * cubic_members, atol=1e-5)
    assert jnp.allclose(banzhaf, base + 1.5 / 4 * cubic_members, atol=1e-5)


@pytest.mark.parametrize(("index_type", "weight_factory"), [(SII, sii_weight), (BII, bii_weight)])
def test_exact_base_interactions_match_brute_force(index_type, weight_factory):
    explanation = ExactExplainer(game_from(cubic_from_masks), index_type(order=2)).explain()
    for player in range(N_PLAYERS):
        expected = brute_force_base_interaction(cubic_from_masks, (player,), weight_factory(1))
        assert jnp.allclose(explanation((player,)), expected, atol=1e-4)
    for pair in combinations(range(N_PLAYERS), 2):
        expected = brute_force_base_interaction(cubic_from_masks, pair, weight_factory(2))
        assert jnp.allclose(explanation(pair), expected, atol=1e-4)


def test_exact_stii_matches_derivatives_and_top_order_weights():
    explanation = ExactExplainer(game_from(cubic_from_masks), STII(order=3)).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(explanation((player,)), WEIGHTS[player], atol=1e-4)
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(explanation((left, right)), PAIRS[left, right], atol=1e-4)
    for triple in combinations(range(N_PLAYERS), 3):
        expected = 1.5 if triple == (0, 1, 2) else 0.0
        assert jnp.allclose(explanation(triple), expected, atol=1e-4)


def test_exact_fsii_recovers_the_moebius_basis_of_quadratic_games():
    explanation = ExactExplainer(game_from(quadratic_from_masks), FSII(order=2)).explain()
    assert jnp.allclose(order_one(explanation), WEIGHTS, atol=1e-4)
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(explanation((left, right)), PAIRS[left, right], atol=1e-4)


def test_exact_fsii_of_order_one_is_the_shapley_value():
    game = game_from(cubic_from_masks)
    fsii = ExactExplainer(game, FSII(order=1)).explain()
    shapley = ExactExplainer(game_from(cubic_from_masks), SV()).explain()
    assert jnp.allclose(order_one(fsii), order_one(shapley), atol=1e-4)


def test_exact_shapley_values_are_efficient():
    game = game_from(cubic_from_masks)
    explanation = ExactExplainer(game, SV()).explain()
    grand = cubic_from_masks(jnp.ones(N_PLAYERS, dtype=jnp.float32))
    empty = cubic_from_masks(jnp.zeros(N_PLAYERS, dtype=jnp.float32))
    assert jnp.allclose(jnp.sum(order_one(explanation)), grand - empty, atol=1e-4)
    assert jnp.allclose(explanation.baseline, empty, atol=1e-6)


def test_exact_explainer_agrees_with_permutation_sampling():
    exact = order_one(ExactExplainer(game_from(quadratic_from_masks), SV()).explain())
    sampled = PermutationSampling(game_from(quadratic_from_masks), SV(), random_state=7)
    estimate = order_one(sampled.sample(2 + 3000 * (N_PLAYERS - 1)).explain())
    assert jnp.allclose(estimate, exact, atol=0.05)


def test_exact_indices_on_a_random_game_match_brute_force():
    # a game with Moebius mass on every order pins the full weight profiles,
    # which the structured quadratic and cubic games cannot
    mask_fn = random_table_game()
    sii = ExactExplainer(game_from(mask_fn), SII(order=3)).explain()
    bii = ExactExplainer(game_from(mask_fn), BII(order=3)).explain()
    stii = ExactExplainer(game_from(mask_fn), STII(order=2)).explain()
    for triple in [(0, 1, 2), (1, 3, 4)]:
        expected = brute_force_base_interaction(mask_fn, triple, sii_weight(3))
        assert jnp.allclose(sii(triple), expected, atol=1e-4)
        expected = brute_force_base_interaction(mask_fn, triple, bii_weight(3))
        assert jnp.allclose(bii(triple), expected, atol=1e-4)
    for pair in [(0, 1), (2, 4)]:
        expected = brute_force_base_interaction(mask_fn, pair, stii_top_weight(2))
        assert jnp.allclose(stii(pair), expected, atol=1e-4)
    for player in (0, 3):
        expected = discrete_derivative(mask_fn, (player,), ())
        assert jnp.allclose(stii((player,)), expected, atol=1e-4)


def test_game_is_evaluated_once_across_explanations():
    n_rows = []

    def recording(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        n_rows.append(masks.shape[-2])
        return quadratic_from_masks(masks)

    game = CallableGame(fn=recording, n_players=N_PLAYERS)
    explainer = ExactExplainer(game, SII(order=2))
    explainer.explain()
    explainer.explain()
    assert n_rows == [2**N_PLAYERS]


def test_unsupported_indices_are_rejected():
    with pytest.raises(TypeError, match=r"shapiq\.SII\(order=2\) instead of 'SII'"):
        ExactExplainer(game_from(quadratic_from_masks), "SII")

    class WeightlessIndex:
        name = "SII"
        order = 1
        order_semantics = "coverage"
        includes_empty_interaction = False

    with pytest.raises(TypeError, match="does not support"):
        ExactExplainer(game_from(quadratic_from_masks), WeightlessIndex())
    with pytest.raises(ValueError, match="order"):
        ExactExplainer(game_from(quadratic_from_masks), SII(order=N_PLAYERS + 1))
    with pytest.raises(TypeError):
        SV(order=1)  # the Shapley value has no order freedom


def test_index_objects_expose_order_and_semantics():
    assert SV().order == 1
    assert SII().order == 2
    assert SII(order=3).order_semantics == "coverage"
    assert STII(order=3).order_semantics == "identity"
    assert FSII().order_semantics == "identity"
    with pytest.raises(ValueError, match="order"):
        SII(order=0)
