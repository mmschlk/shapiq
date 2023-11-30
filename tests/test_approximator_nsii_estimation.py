"""Tests the approximiation of nSII values with PermutationSamplingSII and ShapIQ."""
import numpy as np
import pytest

from approximator import convert_nsii_into_one_dimension, transforms_sii_to_nsii
from shapiq import DummyGame, PermutationSamplingSII, ShapIQ


@pytest.mark.parametrize(
    "sii_approximator, nsii_approximator",
    [
        (
            PermutationSamplingSII(7, 2, "SII", False, random_state=42),
            PermutationSamplingSII(7, 2, "nSII", False, random_state=42),
        ),
        (ShapIQ(7, 2, "SII", False, random_state=42), ShapIQ(7, 2, "nSII", False, random_state=42)),
    ],
)
def test_nsii_estimation(sii_approximator, nsii_approximator):
    """Tests the approximation of nSII values with PermutationSamplingSII and ShapIQ."""
    n = 7
    max_order = 2
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    # sii_approximator = PermutationSamplingSII(n, max_order, "SII", False, random_state=42)
    sii_estimates = sii_approximator.approximate(1_000, game, batch_size=None)
    # nsii_approximator = PermutationSamplingSII(n, max_order, "nSII", False, random_state=42)
    nsii_estimates = nsii_approximator.approximate(1_000, game, batch_size=None)
    assert sii_estimates != nsii_estimates
    assert nsii_estimates.index == "nSII"

    n_sii_transformed = nsii_approximator.transforms_sii_to_nsii(sii_estimates)
    assert n_sii_transformed.index == "nSII"
    assert n_sii_transformed == nsii_estimates  # check weather transform and estimation are equal

    # nSII values for player 1 and 2 should be approximately 0.1429 and the interaction 1.0
    assert nsii_estimates[(1,)] == pytest.approx(0.1429, 0.4)
    assert nsii_estimates[(2,)] == pytest.approx(0.1429, 0.4)
    assert nsii_estimates[(1, 2)] == pytest.approx(1.0, 0.2)

    # check efficiency
    efficiency = np.sum(nsii_estimates.values)
    assert efficiency == pytest.approx(2.0, 0.01)

    # check one dim transform
    pos_nsii_values, neg_nsii_values = convert_nsii_into_one_dimension(nsii_estimates)
    assert pos_nsii_values.shape == (n,) and neg_nsii_values.shape == (n,)
    assert np.all(pos_nsii_values >= 0) and np.all(neg_nsii_values <= 0)
    sum_of_both = np.sum(pos_nsii_values) + np.sum(neg_nsii_values)
    assert sum_of_both == pytest.approx(efficiency, 0.01)
    assert sum_of_both != pytest.approx(0.0, 0.01)

    with pytest.raises(ValueError):
        _ = convert_nsii_into_one_dimension(sii_estimates)

    # check transforms_sii_to_nsii function
    transformed = transforms_sii_to_nsii(sii_estimates)
    assert transformed.index == "nSII"
    transformed = transforms_sii_to_nsii(sii_estimates.values, approximator=sii_approximator)
    assert isinstance(transformed, np.ndarray)
    transformed = transforms_sii_to_nsii(sii_estimates.values, n=n, max_order=max_order)
    assert isinstance(transformed, np.ndarray)
    with pytest.raises(ValueError):
        _ = transforms_sii_to_nsii(sii_estimates.values)
