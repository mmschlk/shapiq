from shapiq.games.soum import SOUM
from shapiq.approximator.moebius_converter import MoebiusConverter
from shapiq.exact_computer import ExactComputer
import numpy as np


def test_exact_computer_on_soum():
    for i in range(100):
        n = np.random.randint(low=2, high=10)
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

        moebius_transform = exact_computer.moebius_transform()
        # Assert equality with ground truth Möbius coefficients from SOUM
        assert np.sum((moebius_transform - soum.moebius_coefficients).values ** 2) < 10e-7

        # Compare ground truth via MoebiusConvert with exact computation of ExactComputer
        shapley_interactions_gt = {}
        shapley_interactions_exact = {}
        for index in ["STII", "k-SII", "FSII"]:
            shapley_interactions_gt[index] = moebius_converter.moebius_to_shapley_interaction(
                order, index
            )
            shapley_interactions_exact[index] = exact_computer.shapley_interaction(order, index)
            # Check equality with ground truth calculations from SOUM
            assert (
                np.sum(
                    (shapley_interactions_exact[index] - shapley_interactions_gt[index]).values ** 2
                )
                < 10e-7
            )

        index = "JointSV"
        shapley_generalized_values = exact_computer.shapley_generalized_value(
            order=order, index=index
        )
        # Assert efficiency
        assert (np.sum(shapley_generalized_values.values) - predicted_value) ** 2 < 10e-7

        index = "kADD-SHAP"
        shapley_interactions_exact[index] = exact_computer.shapley_interaction(order, index)

        base_interaction_indices = ["SII", "BII", "CHII"]
        base_interactions = {}
        for base_index in base_interaction_indices:
            base_interactions[base_index] = exact_computer.shapley_base_interaction(
                order=order, index=base_index
            )

        base_gv_indices = ["SGV", "BGV", "CHGV"]
        base_gv = {}
        for base_gv_index in base_gv_indices:
            base_gv[base_gv_index] = exact_computer.base_generalized_value(
                order=order, index=base_gv_index
            )

        probabilistic_values_indices = ["SV", "BV"]
        probabilistic_values = {}
        for pv_index in probabilistic_values_indices:
            probabilistic_values[pv_index] = exact_computer.probabilistic_value(index=pv_index)

        # Assert efficiency for SV
        assert (np.sum(probabilistic_values["SV"].values) - predicted_value) ** 2 < 10e-7