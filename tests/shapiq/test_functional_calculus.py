"""Tests for coalition functionals and the derived Monte Carlo estimator."""

from __future__ import annotations

from itertools import combinations
from math import comb

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    BGV,
    BII,
    CHII,
    EGV,
    FSII,
    KSII,
    SGV,
    SII,
    STII,
    SV,
    CallableGame,
    CoalitionSizeSampler,
    CoMoebius,
    DenseCoalitionArray,
    ExactExplainer,
    JointSV,
    Moebius,
    MonteCarlo,
    define_cardinal_index,
    define_generalized_value,
    derive_functional,
    iter_interactions,
)

N_PLAYERS = 5


def random_table_game(seed=42):
    rng = np.random.default_rng(seed)
    table = rng.normal(size=2**N_PLAYERS)

    def mask_fn(masks):
        indices = jnp.asarray(masks @ (2.0 ** jnp.arange(N_PLAYERS)), dtype=jnp.int32)
        return jnp.asarray(table, dtype=jnp.float32)[indices]

    return CallableGame(
        fn=lambda c: mask_fn(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def max_disagreement(explanation, reference, order, min_order=1):
    """Return the largest attribution difference over all shared interactions."""
    return max(
        float(jnp.max(jnp.abs(explanation(interaction) - reference(interaction))))
        for interaction in iter_interactions(N_PLAYERS, order, min_order=min_order)
    )


# tolerances are three standard deviations of the measured per-interaction
# estimator spread at this budget; the Moebius transform is heavy-tailed
# because only coalitions inside an interaction carry coefficient mass
CONVERGING_INDICES = [
    (SV(), 0.15),
    (BII(order=2), 0.2),
    (CHII(order=2), 0.25),
    (SII(order=2), 0.25),
    (STII(order=2), 0.25),
    (KSII(order=2), 0.25),
    (Moebius(order=3), 0.65),
    (CoMoebius(order=2), 0.25),
    (SGV(order=2), 0.25),
    (BGV(order=2), 0.25),
    (EGV(order=2), 0.25),
    (JointSV(order=2), 0.25),
]


@pytest.mark.parametrize(
    ("index", "tolerance"),
    CONVERGING_INDICES,
    ids=lambda value: value.name if hasattr(value, "name") else None,
)
def test_montecarlo_converges_to_exact(index, tolerance):
    game = random_table_game()
    exact = ExactExplainer(game, index).explain()
    estimate = MonteCarlo(game, index, random_state=7).sample(4096).explain()
    min_order = getattr(index, "min_interaction_size", 1)
    assert max_disagreement(estimate, exact, exact.order, min_order=min_order) < tolerance


def test_montecarlo_is_unbiased_at_small_budgets():
    game = random_table_game()
    exact = ExactExplainer(game, SII(order=2)).explain()
    estimates = [
        MonteCarlo(game, SII(order=2), random_state=seed).sample(34).explain()
        for seed in range(96)
    ]
    for interaction in iter_interactions(N_PLAYERS, 2, min_order=1):
        mean = float(jnp.mean(jnp.stack([estimate(interaction) for estimate in estimates])))
        assert mean == pytest.approx(float(exact(interaction)), abs=0.2)


def test_anchored_attributions_are_exact_from_the_seed_block():
    game = random_table_game()
    powerset = ExactExplainer(game, Moebius(order=2))
    exact_empty = float(powerset.explain()(()))
    moebius = MonteCarlo(game, Moebius(order=2), random_state=0).sample(8).explain()
    co_moebius = MonteCarlo(game, CoMoebius(order=2), random_state=0).sample(8).explain()
    exact_grand = float(ExactExplainer(game, CoMoebius(order=2)).explain()(()))
    assert float(moebius(())) == pytest.approx(exact_empty, abs=1e-6)
    assert float(co_moebius(())) == pytest.approx(exact_grand, abs=1e-6)


def test_montecarlo_estimates_do_not_depend_on_budget_splits():
    game = random_table_game()
    whole = MonteCarlo(game, SII(order=2), random_state=3).sample(120)
    split = MonteCarlo(game, SII(order=2), random_state=3).sample(7).sample(50).sample(63)
    assert max_disagreement(whole.explain(), split.explain(), 2) < 1e-6


def test_paired_montecarlo_converges_for_the_shapley_value():
    game = random_table_game()
    exact = ExactExplainer(game, SV()).explain()
    estimate = MonteCarlo(game, SV(), random_state=11, paired=True).sample(4096).explain()
    assert max_disagreement(estimate, exact, 1) < 0.1


def test_deduplicated_montecarlo_reuses_stored_values():
    game = random_table_game()
    estimate = MonteCarlo(
        game,
        SII(order=2),
        random_state=5,
        share_samples=True,
        deduplicate=True,
    )
    with pytest.warns(match="no novel coalitions"):
        estimate = estimate.sample(64)
    explanation = estimate.explain()
    exact = ExactExplainer(game, SII(order=2)).explain()
    assert max_disagreement(explanation, exact, 2) < 5.0


def test_derived_sampler_skips_sizes_without_coefficient_mass():
    game = random_table_game()
    approximator = MonteCarlo(game, Moebius(order=2), random_state=1).sample(66)
    masks = jnp.asarray(approximator.state.coalitions.to_dense())
    sampled_sizes = jnp.sum(masks[..., 2:, :], axis=-1)
    assert bool(jnp.all(sampled_sizes >= 1))
    assert bool(jnp.all(sampled_sizes <= 2))


def test_size_sampler_requires_interior_mass():
    weights = jnp.zeros(N_PLAYERS + 1).at[0].set(1.0).at[N_PLAYERS].set(1.0)
    with pytest.raises(ValueError, match="use ExactExplainer"):
        CoalitionSizeSampler(N_PLAYERS, size_weights=weights)


def test_montecarlo_rejects_indices_without_a_functional():
    game = random_table_game()
    with pytest.raises(TypeError, match="MonteCarlo does not support 'FSII'"):
        MonteCarlo(game, FSII(order=2))


def test_defined_cardinal_index_yields_exact_and_sampled_estimators():
    def tilted_weights(n_players, interaction_size):
        free = n_players - interaction_size
        raw = jnp.arange(1, free + 2, dtype=jnp.float32)
        return raw / jnp.sum(raw * jnp.asarray([comb(free, t) for t in range(free + 1)]))

    index = define_cardinal_index("TiltedII", weights=tilted_weights, order=2)
    game = random_table_game()
    exact = ExactExplainer(game, index).explain()
    assert exact.interaction_index == "TiltedII"
    for interaction in [(0,), (1, 3)]:
        expected = _brute_force_cardinal(game, interaction, tilted_weights)
        assert float(exact(interaction)) == pytest.approx(expected, abs=1e-4)
    estimate = MonteCarlo(game, index, random_state=13).sample(4096).explain()
    assert max_disagreement(estimate, exact, 2) < 0.15


def test_defined_generalized_value_yields_exact_and_sampled_estimators():
    def bloc_weights(n_players, interaction_size):
        free = n_players - interaction_size
        return jnp.full(free + 1, 1.0 / (free + 1))

    index = define_generalized_value("MeanBlocGV", weights=bloc_weights, order=2)
    game = random_table_game()
    exact = ExactExplainer(game, index).explain()
    estimate = MonteCarlo(game, index, random_state=17).sample(4096).explain()
    assert exact.interaction_index == "MeanBlocGV"
    assert max_disagreement(estimate, exact, 2) < 0.15


def test_derive_functional_validates_declared_weight_profiles():
    index = define_cardinal_index(
        "Broken",
        weights=lambda n_players, interaction_size: jnp.ones(n_players),
    )
    with pytest.raises(ValueError, match="weight profile"):
        derive_functional(index, N_PLAYERS, 2)


def _brute_force_cardinal(game, interaction, weight_fn):
    def value(subset):
        mask = np.zeros((1, N_PLAYERS), dtype=bool)
        mask[0, list(subset)] = True
        return float(jnp.asarray(game(DenseCoalitionArray(jnp.asarray(mask))))[0])

    weights = np.asarray(weight_fn(N_PLAYERS, len(interaction)))
    others = [player for player in range(N_PLAYERS) if player not in interaction]
    total = 0.0
    for size in range(len(others) + 1):
        for base in combinations(others, size):
            derivative = 0.0
            for length in range(len(interaction) + 1):
                for part in combinations(interaction, length):
                    sign = (-1.0) ** (len(interaction) - length)
                    derivative += sign * value((*base, *part))
            total += float(weights[size]) * derivative
    return total
