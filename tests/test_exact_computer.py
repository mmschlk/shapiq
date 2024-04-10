from shapiq.games.soum import SOUM
from shapiq.approximator.moebius_converter import MoebiusConverter
from shapiq.exact_computer import ExactComputer
import numpy as np


def test_exact_computer_on_soum():
    for i in range(100):
        n = np.random.randint(low=2, high=12)
        N = set(range(n))
        order = np.random.randint(low=1, high=min(n, 5))
        n_basis_games = np.random.randint(low=1, high=100)
        soum = SOUM(n, n_basis_games=1, min_interaction_size=1)

        predicted_value = soum(np.ones(n))[0]
        emptyset_prediction = soum(np.zeros(n))[0]

        # Compute via exactComputer
        exact_computer = ExactComputer(N, soum)

        # Compute via sparse Möbius representation
        moebius_converter = MoebiusConverter(N, soum.moebius_coefficients)

        # Compare ground truth via MoebiusConvert with exact computation of ExactComputer
        shapley_interactions_gt = {}
        shapley_interactions_exact = {}
        for index in ["STII", "k-SII", "FSII"]:
            shapley_interactions_gt[index] = moebius_converter.moebius_to_shapley_interaction(
                order, index
            )
            shapley_interactions_exact[index] = exact_computer.shapley_interaction(order, index)
            assert (
                np.sum(
                    (shapley_interactions_exact[index] - shapley_interactions_gt[index]).values ** 2
                )
                < 10e-7
            )
