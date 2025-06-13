"""Tests the approximiation of k-SII values with PermutationSamplingSII and SHAPIQ."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.approximator import (
    SHAPIQ,
    PermutationSamplingSII,
)
from shapiq.game_theory.aggregation import aggregate_to_one_dimension
from shapiq.games.benchmark import DummyGame


@pytest.mark.parametrize(
    ("sii_approximator", "ksii_approximator"),
    [
        (
            PermutationSamplingSII(7, 2, "SII", top_order=False, random_state=42),
            PermutationSamplingSII(7, 2, "k-SII", top_order=False, random_state=42),
        ),
        (
            SHAPIQ(7, 2, "SII", top_order=False, random_state=42),
            SHAPIQ(7, 2, "k-SII", top_order=False, random_state=42),
        ),
    ],
)
def test_k_sii_estimation(sii_approximator, ksii_approximator):
    """Tests the approximation of k-SII values with PermutationSamplingSII and ShapIQ."""
    n = 7
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    sii_estimates = sii_approximator.approximate(1_000, game)
    ksii_estimates = ksii_approximator.approximate(1_000, game)
    assert sii_estimates != ksii_estimates
    assert ksii_estimates.index == "k-SII"

    k_sii_transformed = ksii_approximator.aggregate_interaction_values(sii_estimates)
    assert k_sii_transformed.index == "k-SII"
    assert k_sii_transformed == ksii_estimates  # check weather transform and estimation are equal

    # k-SII values for player 1 and 2 should be approximately 0.1429 and the interaction 1.0
    assert ksii_estimates[(1,)] == pytest.approx(0.1429, 0.4)
    assert ksii_estimates[(2,)] == pytest.approx(0.1429, 0.4)
    assert ksii_estimates[(1, 2)] == pytest.approx(1.0, 0.2)
    # the sum should be 2.0
    efficiency = np.sum(ksii_estimates.values)
    assert efficiency == pytest.approx(2.0, 0.01)


def test_k_one_dim_aggregate():
    """Tests the aggregation of k-SII values to one dimension."""
    n = 7
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    estimator = SHAPIQ(7, 2, "k-SII", top_order=False, random_state=42)
    k_sii_estimates = estimator.approximate(2**n, game)

    efficiency = np.sum(k_sii_estimates.values)

    # check one dim transform
    pos_values, neg_values = aggregate_to_one_dimension(k_sii_estimates)
    assert pos_values.shape == (n,)
    assert neg_values.shape == (n,)
    assert np.all(pos_values >= 0)
    assert np.all(neg_values <= 0)
    sum_of_both = np.sum(pos_values) + np.sum(neg_values)

    assert sum_of_both == pytest.approx(efficiency, 0.01)
    assert sum_of_both != pytest.approx(0.0, 0.01)
