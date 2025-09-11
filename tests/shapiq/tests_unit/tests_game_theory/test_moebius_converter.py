"""This test module contains all tests regarding the Möbius converter class."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq_games.synthetic.soum import SOUM


def test_soum_moebius_conversion():
    """Test the basic funcitonality of the Möbius converter."""
    for _ in range(10):
        n = np.random.randint(low=2, high=20)
        order = np.random.randint(low=1, high=min(n, 5))
        n_basis_games = np.random.randint(low=1, high=100)

        soum = SOUM(n, n_basis_games=n_basis_games)

        predicted_value = soum(np.ones(n))[0]
        emptyset_prediction = soum(np.zeros(n))[0]

        moebius_converter = MoebiusConverter(soum.moebius_coefficients)

        for index in ["STII", "k-SII", "FSII"]:
            shapley_interactions = moebius_converter(index=index, order=order)
            # Assert efficiency
            assert (np.sum(shapley_interactions.values) - predicted_value) ** 2 < 10e-7
            assert (shapley_interactions[()] - emptyset_prediction) ** 2 < 10e-7
            for v in moebius_converter._computed.values():
                # check that no 0's are in the interaction lookup (except for the empty set, which is the first entry)
                interactions = v.interactions
                assert all(v != 0 for idx, v in enumerate(interactions.values()) if idx > 0)

        # test direct call of Möbius converter
        for index in ["STII", "k-SII", "SII", "FSII"]:
            moebius_converter(index=index, order=order)


@pytest.mark.parametrize("random_state", [10, 19, 20, 21, 23])
def test_soum_moebius_conversion_failing_states(random_state):
    """Test SOUM moebius conversion with specific failing random states."""
    order = 3
    n_basis_games = 1
    n = 7

    soum = SOUM(n, n_basis_games=n_basis_games, random_state=random_state)
    predicted_value = soum(np.ones(n))[0]
    emptyset_prediction = soum(np.zeros(n))[0]

    moebius_converter = MoebiusConverter(soum.moebius_coefficients)

    for index in ["STII", "k-SII", "FSII"]:
        shapley_interactions = moebius_converter(index=index, order=order)
        # Assert efficiency
        assert (np.sum(shapley_interactions.values) - predicted_value) ** 2 < 10e-7
        assert (shapley_interactions[()] - emptyset_prediction) ** 2 < 10e-7
        for v in moebius_converter._computed.values():
            interactions = v.interactions
            # Check that no 0's are in the interaction values (except for empty set)
            non_empty_values = [val for idx, val in enumerate(interactions.values()) if idx > 0]
            assert all(val != 0 for val in non_empty_values), (
                f"Found zero values in non-empty interactions with random state {random_state}"
            )
