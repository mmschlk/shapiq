"""Tests the approximiation of nSII values with PermutationSamplingSII and ShapIQ."""
import numpy as np
import pytest

from approximator import (
    convert_ksii_into_one_dimension,
    transforms_sii_to_ksii,
    PermutationSamplingSII,
    ShapIQ,
)
from games import DummyGame


@pytest.mark.parametrize(
    "sii_approximator, ksii_approximator",
    [
        (
            PermutationSamplingSII(7, 2, "SII", False, random_state=42),
            PermutationSamplingSII(7, 2, "k-SII", False, random_state=42),
        ),
        (
            ShapIQ(7, 2, "SII", False, random_state=42),
            ShapIQ(7, 2, "k-SII", False, random_state=42),
        ),
    ],
)
def test_nsii_estimation(sii_approximator, ksii_approximator):
    """Tests the approximation of k-SII values with PermutationSamplingSII and ShapIQ."""
    n = 7
    max_order = 2
    interaction = (1, 2)
    game = DummyGame(n, interaction)
    # sii_approximator = PermutationSamplingSII(n, max_order, "SII", False, random_state=42)
    sii_estimates = sii_approximator.approximate(1_000, game, batch_size=None)
    # nsii_approximator = PermutationSamplingSII(n, max_order, "kSII", False, random_state=42)
    ksii_estimates = ksii_approximator.approximate(1_000, game, batch_size=None)
    assert sii_estimates != ksii_estimates
    assert ksii_estimates.index == "k-SII"

    k_sii_transformed = ksii_approximator.transforms_sii_to_ksii(sii_estimates)
    assert k_sii_transformed.index == "k-SII"
    assert k_sii_transformed == ksii_estimates  # check weather transform and estimation are equal

    # nSII values for player 1 and 2 should be approximately 0.1429 and the interaction 1.0
    assert ksii_estimates[(1,)] == pytest.approx(0.1429, 0.4)
    assert ksii_estimates[(2,)] == pytest.approx(0.1429, 0.4)
    assert ksii_estimates[(1, 2)] == pytest.approx(1.0, 0.2)

    # check efficiency
    efficiency = np.sum(ksii_estimates.values)
    assert efficiency == pytest.approx(2.0, 0.01)

    # check one dim transform
    pos_ksii_values, neg_ksii_values = convert_ksii_into_one_dimension(ksii_estimates)
    assert pos_ksii_values.shape == (n,) and neg_ksii_values.shape == (n,)
    assert np.all(pos_ksii_values >= 0) and np.all(neg_ksii_values <= 0)
    sum_of_both = np.sum(pos_ksii_values) + np.sum(neg_ksii_values)
    assert sum_of_both == pytest.approx(efficiency, 0.01)
    assert sum_of_both != pytest.approx(0.0, 0.01)

    with pytest.raises(ValueError):
        _ = convert_ksii_into_one_dimension(sii_estimates)

    # check transforms_sii_to_nsii function
    transformed = transforms_sii_to_ksii(sii_estimates)
    assert transformed.index == "k-SII"
    transformed = transforms_sii_to_ksii(sii_estimates.values, approximator=sii_approximator)
    assert isinstance(transformed, np.ndarray)
    transformed = transforms_sii_to_ksii(sii_estimates.values, n=n, max_order=max_order)
    assert isinstance(transformed, np.ndarray)
    with pytest.raises(ValueError):
        _ = transforms_sii_to_ksii(sii_estimates.values)
