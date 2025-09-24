"""This test module contains all tests regarding the Möbius converter class."""

from __future__ import annotations

import numpy as np

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

        shapley_interactions = {}
        for index in ["STII", "k-SII", "FSII"]:
            shapley_interactions[index] = moebius_converter(index=index, order=order)
            # Assert efficiency
            assert (np.sum(shapley_interactions[index].values) - predicted_value) ** 2 < 10e-7
            assert (shapley_interactions[index][()] - emptyset_prediction) ** 2 < 10e-7

        # test direct call of Möbius converter
        for index in ["STII", "k-SII", "SII", "FSII", "Moebius"]:
            moebius_converter(index=index, order=order)
