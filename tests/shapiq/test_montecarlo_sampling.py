"""Tests for the SHAP-IQ and SVARM-IQ counts-times-law estimators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    FSII,
    SHAPIQ,
    SII,
    STII,
    SV,
    SVARMIQ,
    CallableGame,
    ExactExplainer,
    InsufficientSamplesError,
    SamplingStallWarning,
)

N_PLAYERS = 7


def structured_game():
    rng = np.random.default_rng(3)
    weights = jnp.asarray(rng.normal(size=N_PLAYERS), dtype=jnp.float32)
    pairs = jnp.asarray(rng.normal(size=(N_PLAYERS, N_PLAYERS)) * 0.4, dtype=jnp.float32)

    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        redundancy = 1.1 * masks[..., [1, 3, 5]].max(axis=-1)
        quadratic = masks @ weights + 0.5 * jnp.einsum(
            "...i,ij,...j->...", masks, pairs, masks,
        )
        return quadratic + redundancy

    return CallableGame(fn=fn, n_players=N_PLAYERS)


def max_error(estimate, exact):
    errors = [
        abs(float(estimate[tuple(term)]) - float(exact[tuple(term)]))
        for term in exact.terms
        if len(term) >= 1
    ]
    return max(errors)


@pytest.mark.parametrize("policy_type", [SHAPIQ, SVARMIQ])
@pytest.mark.parametrize("index", [SV(), SII(order=2)])
def test_estimates_converge_to_the_exact_index(policy_type, index):
    game = structured_game()
    exact = ExactExplainer(game, index).estimate()
    policy = policy_type(game, index, random_state=0)
    assert max_error(policy.estimate(3000), exact) < 0.35
    assert policy.estimate(3000)[()] == pytest.approx(float(exact[()]))


def test_the_stratified_variant_estimates_tighter_here():
    game = structured_game()
    index = SII(order=2)
    exact = ExactExplainer(game, index).estimate()
    shapiq_error = max_error(SHAPIQ(game, index, random_state=0).estimate(3000), exact)
    svarmiq_error = max_error(SVARMIQ(game, index, random_state=0).estimate(3000), exact)
    assert svarmiq_error < shapiq_error


def test_taylor_indices_ride_the_cardinal_capability():
    game = structured_game()
    index = STII(order=2)
    exact = ExactExplainer(game, index).estimate()
    estimate = SVARMIQ(game, index, random_state=0).estimate(3000)
    assert max_error(estimate, exact) < 0.5


@pytest.mark.parametrize("policy_type", [SHAPIQ, SVARMIQ])
def test_budget_splits_never_change_the_estimate(policy_type):
    game = structured_game()
    policy = policy_type(game, SII(order=2), random_state=5)
    whole = policy.estimate(300)
    split = policy.refine(policy.estimate(130), 170)
    assert whole.terms == split.terms
    assert bool(jnp.all(jnp.asarray(whole.coefficients) == jnp.asarray(split.coefficients)))
    assert whole.bank == split.bank


def test_deduplication_reuses_values_and_banks_when_novelty_runs_dry():
    # 7 players hold only 112 distinct interior coalitions at order 2, so a
    # deduplicating run charges less than the budget and stalls with the
    # remainder banked; duplicates still ride the stream as counts
    game = structured_game()
    index = SII(order=2)
    exact = ExactExplainer(game, index).estimate()
    policy = SVARMIQ(game, index, random_state=2, deduplicate=True)
    with pytest.warns(SamplingStallWarning):
        estimate = policy.estimate(600)
    assert estimate.spent < 600
    assert estimate.bank > 0
    assert estimate.evidence.n_samples > estimate.spent  # duplicates stitched in
    # sanity, not convergence: the estimator stays sound on a stalled stream
    assert max_error(estimate, exact) < 1.0


def test_policies_refuse_each_others_carries():
    game = structured_game()
    carry = SHAPIQ(game, SII(order=2), random_state=0).estimate(100)
    with pytest.raises(ValueError, match="cannot be continued"):
        SVARMIQ(game, SII(order=2), random_state=0).refine(carry, 100)


def test_non_cardinal_indices_get_the_teaching_error():
    with pytest.raises(TypeError, match="discrete-derivative"):
        SHAPIQ(structured_game(), FSII(order=2))


def test_estimates_below_one_sampled_unit_carry_the_shortfall():
    game = structured_game()
    policy = SHAPIQ(game, SII(order=2), random_state=0)
    carry = policy.estimate(policy.min_budget - 1)
    assert not carry.ready
    assert carry.bank + carry.evidence.n_samples == policy.min_budget - 1
    with pytest.raises(InsufficientSamplesError, match="at least"):
        carry[(0, 1)]
    grown = policy.refine(carry, 1)
    assert grown.ready


def test_fully_deterministic_regimes_point_to_the_exact_explainer():
    small = CallableGame(
        fn=lambda c: jnp.sum(jnp.asarray(c.to_dense(), dtype=jnp.float32), axis=-1),
        n_players=3,
    )
    with pytest.raises(ValueError, match="nothing left to sample"):
        SHAPIQ(small, SII(order=2))
