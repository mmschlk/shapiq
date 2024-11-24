"""This test module tests the ExactComputer class."""
import math

import numpy as np
import pytest

from shapiq import powerset
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
            print(shapley_interactions_exact[index].values)

            print(shapley_interactions_gt[index].values)

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
    [("ELC", 1)],
)
def test_exact_elc_computer_call(index, order):
    """Tests the call function for the ExactComputer."""
    n = 5
    soum = SOUM(n, n_basis_games=10, normalize=True)
    exact_computer = ExactComputer(n_players=n, game_fun=soum)
    interaction_values = exact_computer(index=index, order=order)
    if order is None:
        order = n
    assert interaction_values is not None  # should return something
    assert interaction_values.max_order == order  # order should be the same
    assert interaction_values.min_order == 1  # ELC only has singleton values
    assert interaction_values.index == index  # index should be the same
    assert interaction_values.baseline_value == 0  # ELC needs baseline_value zero
    assert interaction_values.estimated is False  # nothing should be estimated
    assert interaction_values.values is not None  # values should be computed
    assert exact_computer._elc_stability_subsidy is not None  # ELC should have stored subsidy


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
        ("FBII", 2),
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

@pytest.fixture
def original_game():
    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0, 0.2, -0.1, -0.9, 0]
        second_order_coefficients = np.asarray([[0, 0.4, 0, 0, 0], # interaction btw 0, 1; 1, 3 and 2, 4
                                                [0, 0, 0, 0.3, 0],
                                                [0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0]])
        def _interaction(arr: np.ndarray): #dtype bool
            outer = np.outer(arr, arr)
            interaction_array = second_order_coefficients.copy()
            interaction_array[~outer] = 0
            return np.sum(interaction_array)

        value = np.sum(fist_order_coefficients * x_as_float, axis=1)
        interaction_addition = np.apply_along_axis(_interaction, axis=1, arr=X)
        return value + interaction_addition
    return _game_fun

#(fails for [CHII-2] bc empty set is nan)
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
        ("FBII", 2),
        ("JointSV", 2),
        ("kADD-SHAP", 2),
        ("SII", None),
    ],
)
def test_permutation_symmetry(index, order, original_game):
    n = 5
    if order is None:
        order = n
    permutation = (4, 1, 3, 2, 0) # order = 1, its own inverse
    def permutation_game(X: np.ndarray):
        return original_game((X[:, permutation]))
    exact_computer = ExactComputer(n_players=n, game_fun=original_game)
    interaction_values = exact_computer(index=index, order=order)

    perm_exact_computer = ExactComputer(n_players=n, game_fun=permutation_game)
    perm_interaction_values = perm_exact_computer(index=index, order=order)

    # permutation does not matter
    for coalition, value in interaction_values.dict_values.items():
        perm_coalition = tuple(sorted([permutation[player] for player in coalition]))
        assert (value - perm_interaction_values[perm_coalition]) < 10e-7

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
        ("FBII", 2),
        ("JointSV", 2),
        ("kADD-SHAP", 2),
        ("SII", None),
    ],
)
def test_player_symmetry(index, order):
    n = 5
    if order is None:
        order = n

    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0.4, 0.2, -0.1, -0.9, 0.4]
        second_order_coefficients = np.asarray([[0, 0.4, 0.1, 0, 0], # interaction btw 0, 1; 0, 2; 1, 4; 2, 4
                                                [0, 0, 0, 0, 0.4],
                                                [0, 0, 0, 0, 0.1],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0]])
        def _interaction(arr: np.ndarray): #dtype bool
            outer = np.outer(arr, arr)
            interaction_array = second_order_coefficients.copy()
            interaction_array[~outer] = 0
            return np.sum(interaction_array)

        value = np.sum(fist_order_coefficients * x_as_float, axis=1)
        interaction_addition = np.apply_along_axis(_interaction, axis=1, arr=X)
        return value + interaction_addition

    exact_computer = ExactComputer(n_players=n, game_fun=_game_fun)
    interaction_values = exact_computer(index=index, order=order)

    # symmetry of players with same attribution
    for coalition in powerset(range(n-2)):
        coalition_with_first = (0,) + tuple([player+1 for player in coalition])
        coalition_with_last = tuple([player+1 for player in coalition]) + (4,)
        #print(f"{interaction_values[coalition_with_first]} for {coalition_with_first}")
        assert (interaction_values[coalition_with_first] - interaction_values[coalition_with_last]) < 10e-7


@pytest.mark.parametrize(
    "index, order",
    [
        ("SV", 1),
        ("BV", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("Co-Moebius", 2),
        ("STII", 2),
        ("k-SII", 2),
        ("FSII", 2),
        ("FBII", 2),
        ("kADD-SHAP", 2),
        ("SII", None)
    ],
)
def test_null_player(index, order):
    n = 5
    if order is None:
        order = n

    # game with 0, 4 as null players, has interactions
    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0, 0.2, -0.1, -0.9, 0]
        second_order_coefficients = np.asarray([[0, 0, 0, 0, 0],
                                                [0, 0, 0, 0.3, 0],
                                                [0, 0, 0, 0.4, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0]])
        def _interaction(arr: np.ndarray): #dtype bool
            outer = np.outer(arr, arr)
            interaction_array = second_order_coefficients.copy()
            interaction_array[~outer] = 0
            return np.sum(interaction_array)

        value = np.sum(fist_order_coefficients * x_as_float, axis=1)
        interaction_addition = np.apply_along_axis(_interaction, axis=1, arr=X)
        return value + interaction_addition

    exact_computer = ExactComputer(n_players=n, game_fun=_game_fun)
    interaction_values = exact_computer(index=index, order=order)

    # no attribution for coalitions which include the null players.
    for coalition in powerset(range(n-2)):
        coalition_with_first = (0,) + tuple([player+1 for player in coalition])
        coalition_with_last = tuple([player+1 for player in coalition]) + (4,)
        print(f"{interaction_values[coalition_with_first]} for {coalition_with_first}")
        assert interaction_values[coalition_with_first] < 10e-7
        assert interaction_values[coalition_with_last] < 10e-7

@pytest.mark.parametrize(
    "index, order",
    [
        ("SV", 1),
        ("BV", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("Co-Moebius", 2),
        ("STII", 2),
        ("k-SII", 2),
        ("FSII", 2),
        ("FBII", 2),
        ("kADD-SHAP", 2),
        ("SII", None)
    ],
)
def test_no_artefact_interaction(index, order):
    n = 5
    if order is None:
        order = n

    # game without interactions
    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0, 0.2, -0.1, -0.9, 0]
        return np.sum(fist_order_coefficients * x_as_float, axis=1)


    exact_computer = ExactComputer(n_players=n, game_fun=_game_fun)
    interaction_values = exact_computer(index=index, order=order)

    for coalition, value in interaction_values.dict_values.items():
        if len(coalition) > 1:
            assert value < 10e-7

@pytest.mark.parametrize(
    "index, order",
    [
        ("SGV", 2),
        ("BGV", 2),
        ("CHGV", 2),
        ("EGV", 2),
        ("IGV", 2),
        ("JointSV", 2),
    ],
)
def test_generalized_null_player(index, order):
    # implicit in above test for the rest of the indices
    n = 5
    if order is None:
        order = n
    # game with 0, 4 as null players, has interactions
    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0, 0.2, -0.1, -0.9, 0]
        second_order_coefficients = np.asarray([[0, 0, 0, 0, 0],
                                                [0, 0, 0, 0.3, 0],
                                                [0, 0, 0, 0.4, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0]])
        def _interaction(arr: np.ndarray): #dtype bool
            outer = np.outer(arr, arr)
            interaction_array = second_order_coefficients.copy()
            interaction_array[~outer] = 0
            return np.sum(interaction_array)

        value = np.sum(fist_order_coefficients * x_as_float, axis=1)
        interaction_addition = np.apply_along_axis(_interaction, axis=1, arr=X)
        return value + interaction_addition

    exact_computer = ExactComputer(n_players=n, game_fun=_game_fun)
    interaction_values = exact_computer(index=index, order=order)

    # no attribution for coalitions consisting of the null players.
    assert interaction_values[(0, 4)] < 10e-7
    assert interaction_values[(0,)] < 10e-7
    assert interaction_values[(4,)] < 10e-7
