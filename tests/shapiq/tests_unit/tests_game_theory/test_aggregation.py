"""Tests the approximiation of k-SII values with PermutationSamplingSII and SHAPIQ."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.approximator import (
    SHAPIQ,
    PermutationSamplingSII,
)
from shapiq.game_theory.aggregation import (
    _project_onto_lower_orders,
    aggregate_base_attributions,
    aggregate_to_one_dimension,
)
from shapiq_games.synthetic import DummyGame


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


# three order-8 interactions over a wide feature space. Feature ids this large push
# ``(max_feature + 2) ** 8`` past the int64 subset codes of _project_onto_lower_orders, so
# aggregating them takes the loop fallback.
LARGE_ORDER_8_INTERACTIONS = {
    (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007): 1.5,
    (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1008): -0.5,  # shares 7 features with the first
    (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007): 2.0,  # disjoint from both others
}


@pytest.mark.parametrize("per_instance", [False, True])
def test_aggregation_falls_back_to_loop_on_large_feature_ids(per_instance):
    """Tests that the aggregation stays correct when the vectorized projection cannot be used.

    :func:`_project_onto_lower_orders` returns ``None`` for feature ids that overflow its int64
    subset codes, and :func:`aggregate_base_attributions` then falls back to the loop. The same
    interactions relabelled to compact ids do fit the codes, so the vectorized result is the
    reference for the fallback.
    """
    order = 8
    # the aggregation is linear, so a value may also be one entry per explained instance
    weights = np.array([1.0, 2.0, -1.0]) if per_instance else 1.0
    interactions = {key: value * weights for key, value in LARGE_ORDER_8_INTERACTIONS.items()}

    # relabel the 10 distinct features to ids 0..9, which keeps the codes in range
    features = sorted({feature for key in interactions for feature in key})
    compact = {feature: position for position, feature in enumerate(features)}
    compact_items = [
        (tuple(compact[feature] for feature in key), value) for key, value in interactions.items()
    ]

    assert _project_onto_lower_orders(list(interactions.items()), order) is None  # fallback needed
    reference = _project_onto_lower_orders(compact_items, order)
    assert reference is not None  # compact ids: the vectorized path applies

    aggregated, index, min_order = aggregate_base_attributions(
        interactions, index="SII", order=order, min_order=1, baseline_value=0.0
    )
    assert index == "k-SII"
    assert min_order == 0

    # the disjoint interaction gets no contributions from the other two, so its subsets carry
    # exactly bernoulli(8 - |T|) * 2.0, with the odd bernoulli numbers B_3, B_5, B_7 zeroing
    # out the subsets of size 5, 3 and 1
    disjoint = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007)
    np.testing.assert_allclose(aggregated[disjoint], 2.0 * weights)
    np.testing.assert_allclose(aggregated[disjoint[:7]], -0.5 * 2.0 * weights)
    np.testing.assert_allclose(aggregated[disjoint[:6]], 2.0 / 6 * weights)
    np.testing.assert_allclose(aggregated[disjoint[:4]], -2.0 / 30 * weights)
    np.testing.assert_allclose(aggregated[disjoint[:2]], 2.0 / 42 * weights)
    for size in (1, 3, 5):
        assert disjoint[:size] not in aggregated

    # and the full result, overlapping interactions included, matches the vectorized path
    assert aggregated[()] == 0.0
    fallback = {
        tuple(compact[feature] for feature in key): value
        for key, value in aggregated.items()
        if len(key) > 0
    }
    assert fallback.keys() == reference.keys()
    for key, value in reference.items():
        np.testing.assert_allclose(fallback[key], value)
