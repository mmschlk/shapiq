"""Tests for the extended exact index families and declared generalizations."""

from __future__ import annotations

from itertools import combinations
from math import comb

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
    WeightedFBII,
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


def random_table_game():
    rng = np.random.default_rng(42)
    table = rng.normal(size=2**N_PLAYERS)

    def mask_fn(masks):
        indices = jnp.asarray(masks @ (2.0 ** jnp.arange(N_PLAYERS)), dtype=jnp.int32)
        return jnp.asarray(table, dtype=jnp.float32)[indices]

    return mask_fn


def order_one(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


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


def brute_force_generalized_value(mask_fn, interaction, weight_of_t):
    others = [p for p in range(N_PLAYERS) if p not in interaction]
    total = 0.0
    for size in range(len(others) + 1):
        for base in combinations(others, size):
            gain = float(mask_fn(subset_mask((*base, *interaction)))) - float(
                mask_fn(subset_mask(base)),
            )
            total += weight_of_t(size) * gain
    return total


def total_attribution(explanation, order):
    total = 0.0
    for size in range(1, order + 1):
        for interaction in combinations(range(N_PLAYERS), size):
            total += float(explanation(interaction))
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


def chii_weight(interaction_size):
    def weight(t):
        return interaction_size / ((interaction_size + t) * comb(N_PLAYERS, interaction_size + t))

    return weight


GENERALIZING_INDICES = [
    SII,
    BII,
    CHII,
    STII,
    KSII,
    FSII,
    FBII,
    WeightedFBII,
    KADDSHAP,
    SGV,
    BGV,
    CHGV,
    JointSV,
]


@pytest.mark.parametrize("index_type", GENERALIZING_INDICES)
def test_declared_generalizations_hold_at_order_one(index_type):
    mask_fn = random_table_game()
    index = index_type(order=1)
    value = index.generalizes
    assert value is not None
    restricted = ExactExplainer(game_from(mask_fn), index).explain()
    reference = ExactExplainer(game_from(mask_fn), value).explain()
    assert jnp.allclose(order_one(restricted), order_one(reference), atol=1e-4)


def test_values_and_transforms_declare_no_generalization():
    for index in (SV(), BV(), Moebius(), CoMoebius(), IGV(), EGV()):
        assert index.generalizes is None


def test_exact_chii_matches_brute_force():
    mask_fn = random_table_game()
    explanation = ExactExplainer(game_from(mask_fn), CHII(order=2)).explain()
    for player in range(N_PLAYERS):
        expected = brute_force_base_interaction(mask_fn, (player,), chii_weight(1))
        assert jnp.allclose(explanation((player,)), expected, atol=1e-4)
    for pair in combinations(range(N_PLAYERS), 2):
        expected = brute_force_base_interaction(mask_fn, pair, chii_weight(2))
        assert jnp.allclose(explanation(pair), expected, atol=1e-4)


def test_exact_moebius_recovers_interaction_masses():
    explainer = ExactExplainer(game_from(cubic_from_masks), Moebius())
    assert explainer.order == N_PLAYERS
    explanation = explainer.explain()
    assert jnp.allclose(explanation.baseline, 0.0, atol=1e-6)
    for player in range(N_PLAYERS):
        assert jnp.allclose(explanation((player,)), WEIGHTS[player], atol=1e-4)
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(explanation(pair), PAIRS[pair], atol=1e-4)
    for triple in combinations(range(N_PLAYERS), 3):
        expected = 1.5 if triple == (0, 1, 2) else 0.0
        assert jnp.allclose(explanation(triple), expected, atol=1e-4)
    for quadruple in combinations(range(N_PLAYERS), 4):
        assert jnp.allclose(explanation(quadruple), 0.0, atol=1e-4)


def test_exact_co_moebius_matches_derivatives_at_the_complement():
    mask_fn = random_table_game()
    explanation = ExactExplainer(game_from(mask_fn), CoMoebius(order=2)).explain()
    grand = float(mask_fn(subset_mask(range(N_PLAYERS))))
    empty = float(mask_fn(subset_mask(())))
    assert jnp.allclose(explanation.baseline, empty, atol=1e-6)
    assert jnp.allclose(explanation(()), grand - empty, atol=1e-4)
    for interaction in [(1,), (4,), (0, 2), (3, 4)]:
        complement = tuple(p for p in range(N_PLAYERS) if p not in interaction)
        expected = discrete_derivative(mask_fn, interaction, complement)
        assert jnp.allclose(explanation(interaction), expected, atol=1e-4)


def test_exact_ksii_aggregates_sii_and_is_efficient():
    mask_fn = random_table_game()
    ksii = ExactExplainer(game_from(mask_fn), KSII(order=2)).explain()
    sii = ExactExplainer(game_from(mask_fn), SII(order=2)).explain()
    # top-order interactions have no supersets to aggregate over and stay SII
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(ksii(pair), sii(pair), atol=1e-4)
    # players absorb their pair interactions scaled by the Bernoulli number -1/2
    for player in range(N_PLAYERS):
        pairs = sum(
            float(sii(tuple(sorted((player, other)))))
            for other in range(N_PLAYERS)
            if other != player
        )
        assert jnp.allclose(ksii((player,)), float(sii((player,))) - 0.5 * pairs, atol=1e-4)
    grand = float(mask_fn(subset_mask(range(N_PLAYERS))))
    empty = float(mask_fn(subset_mask(())))
    assert jnp.allclose(total_attribution(ksii, 2), grand - empty, atol=1e-3)
    assert jnp.allclose(ksii.baseline, empty, atol=1e-4)


def test_exact_fbii_recovers_the_moebius_basis_of_quadratic_games():
    explanation = ExactExplainer(game_from(quadratic_from_masks), FBII(order=2)).explain()
    assert jnp.allclose(order_one(explanation), WEIGHTS, atol=1e-4)
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(explanation((left, right)), PAIRS[left, right], atol=1e-4)
    assert jnp.allclose(explanation(()), 0.0, atol=1e-4)


def test_exact_kadd_shap_on_quadratic_games_yields_shapley_values_and_pair_masses():
    explanation = ExactExplainer(game_from(quadratic_from_masks), KADDSHAP(order=2)).explain()
    shapley_values = WEIGHTS + 0.5 * jnp.sum(PAIRS, axis=1)
    assert jnp.allclose(order_one(explanation), shapley_values, atol=1e-4)
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(explanation((left, right)), PAIRS[left, right], atol=1e-4)


@pytest.mark.parametrize(
    ("index_type", "weight_factory"),
    [(SGV, sii_weight), (BGV, bii_weight), (CHGV, chii_weight)],
)
def test_exact_generalized_values_match_brute_force(index_type, weight_factory):
    mask_fn = random_table_game()
    explanation = ExactExplainer(game_from(mask_fn), index_type(order=2)).explain()
    for interaction in [(1,), (3,), (0, 2), (2, 4)]:
        expected = brute_force_generalized_value(
            mask_fn,
            interaction,
            weight_factory(len(interaction)),
        )
        assert jnp.allclose(explanation(interaction), expected, atol=1e-4)


def test_exact_internal_and_external_generalized_values_have_closed_forms():
    mask_fn = random_table_game()
    igv = ExactExplainer(game_from(mask_fn), IGV(order=2)).explain()
    egv = ExactExplainer(game_from(mask_fn), EGV(order=2)).explain()
    empty = float(mask_fn(subset_mask(())))
    grand = float(mask_fn(subset_mask(range(N_PLAYERS))))
    for interaction in [(0,), (4,), (2, 3), (0, 1)]:
        joined = float(mask_fn(subset_mask(interaction)))
        assert jnp.allclose(igv(interaction), joined - empty, atol=1e-4)
        complement = tuple(p for p in range(N_PLAYERS) if p not in interaction)
        left_out = float(mask_fn(subset_mask(complement)))
        assert jnp.allclose(egv(interaction), grand - left_out, atol=1e-4)


def test_exact_jointsv_higher_orders_stay_efficient():
    mask_fn = random_table_game()
    joint = ExactExplainer(game_from(mask_fn), JointSV(order=2)).explain()
    grand = float(mask_fn(subset_mask(range(N_PLAYERS))))
    empty = float(mask_fn(subset_mask(())))
    assert jnp.allclose(total_attribution(joint, 2), grand - empty, atol=1e-3)
    assert jnp.allclose(joint.baseline, empty, atol=1e-4)


def test_values_declare_their_singleton_marginal_weights():
    assert jnp.allclose(
        SV().marginal_weights(N_PLAYERS, 1),
        SGV(order=2).marginal_weights(N_PLAYERS, 1),
    )
    assert jnp.allclose(
        BV().marginal_weights(N_PLAYERS, 1),
        BGV(order=2).marginal_weights(N_PLAYERS, 1),
    )
    with pytest.raises(ValueError, match="single players only"):
        SV().marginal_weights(N_PLAYERS, 2)


def test_regression_kernels_declare_their_constraint_structure():
    assert jnp.allclose(FBII(order=2).regression_kernel(4), jnp.ones(5))
    shapley_kernel = FSII(order=2).regression_kernel(N_PLAYERS)
    assert shapley_kernel[0] == 0.0
    assert shapley_kernel[-1] == 0.0
