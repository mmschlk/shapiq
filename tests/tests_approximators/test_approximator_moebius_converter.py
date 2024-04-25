from shapiq.games.benchmark.synthetic.soum import SOUM
from shapiq.moebius_converter import MoebiusConverter
import numpy as np


def test_soum_moebius_conversion():
    for i in range(10):
        n = np.random.randint(low=2, high=20)
        order = np.random.randint(low=1, high=min(n, 5))
        n_basis_games = np.random.randint(low=1, high=100)

        soum = SOUM(n, n_basis_games=n_basis_games)

        predicted_value = soum(np.ones(n))[0]
        emptyset_prediction = soum(np.zeros(n))[0]

        player_set: set = set(range(soum.n_players))
        moebius_converter = MoebiusConverter(player_set, soum.moebius_coefficients)

        shapley_interactions = {}
        for index in ["STII", "k-SII", "FSII"]:
            shapley_interactions[index] = moebius_converter.moebius_to_shapley_interaction(
                index=index, order=order
            )
            # Assert efficiency
            assert (np.sum(shapley_interactions[index].values) - predicted_value) ** 2 < 10e-7
            assert (shapley_interactions[index][tuple()] - emptyset_prediction) ** 2 < 10e-7

        # test direct call of Möbius converter
        for index in ["STII", "k-SII", "SII", "FSII"]:
            moebius_converter(index=index, order=order)
