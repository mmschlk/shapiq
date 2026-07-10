"""Tests for the sampled k-additive Shapley (kADD-SHAP) regression."""

from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import pytest

from shapiq import (
    KADDSHAP,
    SII,
    SV,
    CallableGame,
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


def order_one(explanation):
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)], axis=-1)


@pytest.mark.filterwarnings("ignore::shapiq.errors.SamplingStallWarning")
def test_recovers_the_exact_kadd_fit_once_identified():
    exact = ExactExplainer(game_from(quadratic_from_masks), KADDSHAP(order=2)).explain()
    approximator = Regression(
        game_from(quadratic_from_masks),
        KADDSHAP(order=2),
        random_state=0,
        deduplicate=True,
    )
    explanation = approximator.sample(SEEDS + 26).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(explanation((player,)), exact((player,)), atol=1e-3)
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(explanation(pair), exact(pair), atol=1e-3)


@pytest.mark.filterwarnings("ignore::shapiq.errors.SamplingStallWarning")
def test_order_one_attributions_are_shapley_values():
    # kADD-SHAP preserves the Shapley value at every order; on a 2-additive
    # game the identified fit reproduces the game, so equality is exact
    shapley = order_one(ExactExplainer(game_from(quadratic_from_masks), SV()).explain())
    approximator = Regression(
        game_from(quadratic_from_masks),
        KADDSHAP(order=2),
        random_state=1,
        deduplicate=True,
    )
    estimate = order_one(approximator.sample(SEEDS + 26).explain())
    assert jnp.allclose(estimate, shapley, atol=1e-3)


def test_converges_to_the_exact_kadd_interactions():
    exact = ExactExplainer(game_from(cubic_from_masks), KADDSHAP(order=2)).explain()
    approximator = Regression(game_from(cubic_from_masks), KADDSHAP(order=2), random_state=2)
    explanation = approximator.sample(SEEDS + 6000).explain()
    for player in range(N_PLAYERS):
        assert jnp.allclose(explanation((player,)), exact((player,)), atol=0.1)
    for pair in combinations(range(N_PLAYERS), 2):
        assert jnp.allclose(explanation(pair), exact(pair), atol=0.1)


def test_sampling_is_invariant_to_budget_splits():
    def make():
        return Regression(game_from(cubic_from_masks), KADDSHAP(order=2), random_state=11)

    split = make().sample(7).sample(2).sample(31)
    whole = make().sample(40)
    assert split.state == whole.state
    assert split.sampler.n_pending_samples == whole.sampler.n_pending_samples


def test_deduplication_reproduces_plain_estimates():
    deduplicated = Regression(
        game_from(cubic_from_masks),
        KADDSHAP(order=2),
        random_state=3,
        deduplicate=True,
    ).sample(SEEDS + 24)
    raw_samples = deduplicated.state.n_samples
    plain = Regression(game_from(cubic_from_masks), KADDSHAP(order=2), random_state=3).sample(
        raw_samples,
    )
    assert deduplicated.state == plain.state
    assert jnp.allclose(
        order_one(deduplicated.explain()),
        order_one(plain.explain()),
        atol=1e-6,
    )


@pytest.mark.filterwarnings("ignore::shapiq.errors.SamplingStallWarning")
def test_minimum_budget_and_identification_gate_explanations():
    approximator = Regression(
        game_from(cubic_from_masks),
        KADDSHAP(order=2),
        random_state=0,
        deduplicate=True,
    )
    assert approximator.min_budget == 16  # 15 interaction columns - 1 constraint + 2 seeds
    with pytest.raises(InsufficientSamplesError, match="not yet identified"):
        approximator.sample(SEEDS + 12).explain()
    # the Bernoulli basis identifies more slowly than the interaction design:
    # min_budget is the information-theoretic floor, not a guarantee
    approximator.sample(SEEDS + 26).explain()  # identified: rank 14 of 14


def test_metadata_names_the_index_and_carries_no_intercept():
    approximator = Regression(
        game_from(cubic_from_masks),
        KADDSHAP(order=2),
        random_state=0,
        deduplicate=True,
    )
    explanation = approximator.sample(SEEDS + 24).explain()
    assert explanation.interaction_index == "kADD-SHAP"
    assert explanation.order == 2
    with pytest.raises(KeyError, match="defines no order-0 attribution"):
        explanation(())


def test_entry_gates_keep_teaching():
    with pytest.raises(TypeError, match=r"does not support 'SII'.*KADDSHAP"):
        Regression(game_from(cubic_from_masks), SII(order=2))

    class MyKADD(KADDSHAP):
        pass

    with pytest.raises(TypeError, match="subclasses a supported index"):
        Regression(game_from(cubic_from_masks), MyKADD(order=2))
