"""Tests for the faithful Banzhaf interaction regression approximator."""

from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import pytest

from shapiq import (
    BV,
    FBII,
    CallableGame,
    Estimate,
    ExactExplainer,
    InsufficientSamplesError,
    Regression,
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


def order_one(source):
    read = source.__getitem__ if isinstance(source, Estimate) else source
    return jnp.stack([read((player,)) for player in range(N_PLAYERS)], axis=-1)


def test_recovers_quadratic_games_exactly_once_identified():
    approximator = Regression(
        game_from(quadratic_from_masks),
        FBII(order=2),
        random_state=0,
        deduplicate=True,
    )
    estimate = approximator.estimate(SEEDS + 24)
    assert jnp.allclose(order_one(estimate), WEIGHTS, atol=1e-3)
    for left, right in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(estimate[(left, right)], PAIRS[left, right], atol=1e-3)
    # a 2-additive game's centered fit needs no intercept
    assert jnp.allclose(estimate[()], 0.0, atol=1e-3)
    assert jnp.allclose(estimate.view.baseline, 0.0, atol=1e-5)


def test_order_one_converges_to_the_banzhaf_value():
    exact = order_one(ExactExplainer(game_from(cubic_from_masks), BV()).estimate().view)
    approximator = Regression(game_from(cubic_from_masks), FBII(order=1), random_state=1)
    estimate = order_one(approximator.estimate(SEEDS + 3000))
    assert jnp.allclose(estimate, exact, atol=0.05)


def test_converges_to_the_exact_faithful_banzhaf_interactions():
    exact = ExactExplainer(game_from(cubic_from_masks), FBII(order=2)).estimate().view
    approximator = Regression(game_from(cubic_from_masks), FBII(order=2), random_state=2)
    estimate = approximator.estimate(SEEDS + 6000)
    for player in range(N_PLAYERS):
        assert jnp.allclose(estimate[(player,)], exact((player,)), atol=0.1)
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(estimate[pair], exact(pair), atol=0.1)


def test_sampling_is_invariant_to_budget_splits():
    def make():
        return Regression(game_from(cubic_from_masks), FBII(order=2), random_state=11)

    policy = make()
    split = policy.refine(policy.refine(policy.estimate(7), 2), 31)
    whole = make().estimate(40)
    assert split.evidence == whole.evidence
    assert split.bank == whole.bank


def test_partial_pair_budgets_are_banked():
    def make():
        return Regression(game_from(cubic_from_masks), FBII(order=2), random_state=5)

    complete = make().estimate(SEEDS + 80)
    with_bank = make().estimate(SEEDS + 80 + 1)
    assert with_bank.bank == 1
    assert with_bank.evidence.n_samples == complete.evidence.n_samples
    assert jnp.allclose(
        order_one(with_bank),
        order_one(complete),
        atol=1e-6,
    )


def test_deduplication_reproduces_plain_estimates():
    deduplicated = Regression(
        game_from(cubic_from_masks),
        FBII(order=2),
        random_state=3,
        deduplicate=True,
    ).estimate(SEEDS + 24)
    raw_samples = deduplicated.evidence.n_samples
    plain = Regression(game_from(cubic_from_masks), FBII(order=2), random_state=3).estimate(
        raw_samples,
    )
    assert deduplicated.evidence == plain.evidence
    assert jnp.allclose(
        order_one(deduplicated),
        order_one(plain),
        atol=1e-6,
    )


def test_minimum_budget_and_identification_gate_explanations():
    approximator = Regression(
        game_from(cubic_from_masks),
        FBII(order=2),
        random_state=0,
        deduplicate=True,
    )
    assert approximator.min_budget == 18  # 1 intercept + 15 interaction columns + 2 seeds
    with pytest.raises(InsufficientSamplesError, match="not yet identified"):
        approximator.estimate(SEEDS + 16)[(0,)]
    approximator.estimate(SEEDS + 20)[(0,)]  # identified: rank 16 of 16


def test_metadata_names_the_index():
    approximator = Regression(
        game_from(cubic_from_masks),
        FBII(order=2),
        random_state=0,
        deduplicate=True,
    )
    estimate = approximator.estimate(SEEDS + 24)
    assert estimate.index == FBII(order=2)
    assert estimate.view.order == 2
