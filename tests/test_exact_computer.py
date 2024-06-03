"""This test module tests the ExactComputer class."""

import numpy as np
import pytest

from shapiq.exact import ExactComputer
from shapiq.games.benchmark.synthetic.soum import SOUM
from shapiq.moebius_converter import MoebiusConverter


def test_exact_computer_on_soum():
    for i in range(20):
        n = np.random.randint(low=2, high=10)
        order = np.random.randint(low=1, high=min(n, 5))
        n_basis_games = np.random.randint(low=1, high=100)
        soum = SOUM(n, n_basis_games=n_basis_games)

        predicted_value = soum(np.ones(n))[0]

        # Compute via exactComputer
        exact_computer = ExactComputer(n_players=n, game_fun=soum)

        # Compute via sparse Möbius representation
        moebius_converter = MoebiusConverter(soum.moebius_coefficients)

        moebius_transform = exact_computer.moebius_transform()
        # Assert equality with ground truth Möbius coefficients from SOUM
        assert np.sum((moebius_transform - soum.moebius_coefficients).values ** 2) < 10e-7

        # Compare ground truth via MoebiusConvert with exact computation of ExactComputer
        shapley_interactions_gt = {}
        shapley_interactions_exact = {}
        for index in ["STII", "k-SII", "FSII"]:
            shapley_interactions_gt[index] = moebius_converter.moebius_to_shapley_interaction(
                index=index, order=order
            )
            shapley_interactions_exact[index] = exact_computer.shapley_interaction(
                index=index, order=order
            )
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
        shapley_interactions_exact[index] = exact_computer.shapley_interaction(
            index=index, order=order
        )

        base_interaction_indices = ["SII", "BII", "CHII", "Co-Moebius"]
        base_interactions = {}
        for base_index in base_interaction_indices:
            base_interactions[base_index] = exact_computer.shapley_base_interaction(
                order=order, index=base_index
            )

        base_gv_indices = ["SGV", "BGV", "CHGV", "IGV", "EGV"]
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


@pytest.mark.parametrize(
    "index, order",
    [
        ("SV", 1),
        ("BV", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("Co-Moebius", 2),
        ("SGV", 2),
        ("BGV", 2),
        ("CHGV", 2),
        ("EGV", 2),
        ("IGV", 2),
        ("STII", 2),
        ("k-SII", 2),
        ("FSII", 2),
        ("JointSV", 2),
        ("kADD-SHAP", 2),
        ("SII", None),
    ],
)
def test_exact_computer_call(index, order):
    """Tests the call function for the ExactComputer."""
    n = 5
    soum = SOUM(n, n_basis_games=10)
    exact_computer = ExactComputer(n_players=n, game_fun=soum)
    interaction_values = exact_computer(index=index, order=order)
    if order is None:
        order = n
    assert interaction_values is not None  # should return something
    assert interaction_values.max_order == order  # order should be the same
    assert interaction_values.index == index  # index should be the same
    assert interaction_values.estimated is False  # nothing should be estimated
    assert interaction_values.values is not None  # values should be computed


def test_basic_functions():
    """Tests the basic functions of the ExactComputer."""
    n = 5
    soum = SOUM(n, n_basis_games=10)
    exact_computer = ExactComputer(n_players=n, game_fun=soum)
    isinstance(repr(exact_computer), str)
    isinstance(str(exact_computer), str)
