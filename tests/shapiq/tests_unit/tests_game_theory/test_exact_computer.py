"""This test module tests the ExactComputer class."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import powerset
from shapiq.game_theory.exact import ExactComputer
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq_games.synthetic.soum import SOUM


def test_exact_computer_on_soum():
    """Tests the ExactComputer on the SOUM game."""
    for _ in range(20):
        n = np.random.randint(low=2, high=10)
        order = np.random.randint(low=1, high=min(n, 5))
        n_basis_games = np.random.randint(low=1, high=100)
        soum = SOUM(n, n_basis_games=n_basis_games)

        predicted_value = soum(np.ones(n, dtype=bool))[0]

        # Compute via exactComputer
        exact_computer = ExactComputer(game=soum, n_players=n)

        # Compute via sparse Möbius representation
        moebius_converter = MoebiusConverter(soum.moebius_coefficients)

        moebius_transform = exact_computer.moebius_transform()
        # Assert equality with ground truth Möbius coefficients from SOUM
        assert np.sum((moebius_transform - soum.moebius_coefficients).values ** 2) < 10e-7

        # Compare ground truth via MoebiusConvert with exact computation of ExactComputer
        shapley_interactions_gt = {}
        shapley_interactions_exact = {}
        for index in ("k-SII",):
            shapley_interactions_gt[index] = moebius_converter(index=index, order=order)
            shapley_interactions_exact[index] = exact_computer.shapley_interactions(
                index=index,
                order=order,
            )
            # Check equality with ground truth calculations from SOUM
            assert (
                np.sum(
                    (shapley_interactions_exact[index] - shapley_interactions_gt[index]).values
                    ** 2,
                )
                < 10e-7
            )

        index = "JointSV"
        shapley_generalized_values = exact_computer.shapley_generalized_value(
            order=order,
            index=index,
        )
        # Assert efficiency
        assert (np.sum(shapley_generalized_values.values) - predicted_value) ** 2 < 10e-7

        index = "kADD-SHAP"
        shapley_interactions_exact[index] = exact_computer.shapley_interactions(
            index=index,
            order=order,
        )

        base_interaction_indices = ["SII", "BII", "CHII", "Co-Moebius"]
        base_interactions = {}
        for base_index in base_interaction_indices:
            base_interactions[base_index] = exact_computer.shapley_base_interaction(
                order=order,
                index=base_index,
            )

        base_gv_indices = ["SGV", "BGV", "CHGV", "IGV", "EGV"]
        base_gv = {}
        for base_gv_index in base_gv_indices:
            base_gv[base_gv_index] = exact_computer.base_generalized_value(
                order=order,
                index=base_gv_index,
            )

        probabilistic_values_indices = ["SV", "BV"]
        probabilistic_values = {}
        for pv_index in probabilistic_values_indices:
            probabilistic_values[pv_index] = exact_computer.probabilistic_value(index=pv_index)

        # Assert efficiency for SV
        assert (np.sum(probabilistic_values["SV"].values) - predicted_value) ** 2 < 10e-7


def test_exact_no_n_players():
    """Tests that you can create an ExactComputer without specifying n_players with a Game."""
    n = 5
    soum = SOUM(n, n_basis_games=10)
    exact_computer = ExactComputer(game=soum)
    assert exact_computer.n_players == n


def test_exact_no_n_players_error():
    """Tests that an error is raised if n_players is not specified and no game is provided."""

    def _callable_function(x):
        return np.sum(x, axis=1)

    with pytest.raises(
        ValueError, match="n_players must be specified if game is not a Game object."
    ):
        ExactComputer(game=_callable_function)


@pytest.mark.parametrize(
    ("index", "order"),
    [("ELC", 1)],
)
def test_exact_elc_computer_call(index, order):
    """Tests the call function for the ExactComputer."""
    n = 5
    soum = SOUM(n, n_basis_games=10, normalize=True)
    exact_computer = ExactComputer(game=soum, n_players=n)
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
    ("index", "order"),
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
    exact_computer = ExactComputer(game=soum, n_players=n)
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
    exact_computer = ExactComputer(game=soum, n_players=n)
    isinstance(repr(exact_computer), str)
    isinstance(str(exact_computer), str)


def test_lazy_computation():
    """Tests if the lazy computation (calling without params) works."""
    n = 5
    soum = SOUM(n, n_basis_games=10)
    exact_computer = ExactComputer(game=soum, n_players=n)
    isinstance(repr(exact_computer), str)
    isinstance(str(exact_computer), str)
    sv = exact_computer("SV", 1)
    assert sv.index == "SV"
    assert sv.max_order == 1


@pytest.fixture
def original_game():
    """This fixture returns a game function with interactions."""

    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0, 0.2, -0.1, -0.9, 0]
        second_order_coefficients = np.asarray(
            [
                [0, 0.4, 0, 0, 0],  # interaction btw 0, 1; 1, 3 and 2, 4
                [0, 0, 0, 0.3, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )

        def _interaction(arr: np.ndarray):  # dtype bool
            outer = np.outer(arr, arr)
            interaction_array = second_order_coefficients.copy()
            interaction_array[~outer] = 0
            return np.sum(interaction_array)

        value = np.sum(fist_order_coefficients * x_as_float, axis=1)
        interaction_addition = np.apply_along_axis(_interaction, axis=1, arr=X)
        return value + interaction_addition

    return _game_fun


@pytest.mark.parametrize(
    ("index", "order"),
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
    """This test checks that the values are invariant under permutations of the players."""
    n = 5
    if order is None:
        order = n
    permutation = (4, 1, 3, 2, 0)  # order = 1, its own inverse

    def permutation_game(X: np.ndarray):
        return original_game(X[:, permutation])

    exact_computer = ExactComputer(game=original_game, n_players=n)
    interaction_values = exact_computer(index=index, order=order)

    perm_exact_computer = ExactComputer(game=permutation_game, n_players=n)
    perm_interaction_values = perm_exact_computer(index=index, order=order)

    # permutation does not matter
    for coalition, value in interaction_values.dict_values.items():
        perm_coalition = tuple(sorted([permutation[player] for player in coalition]))
        assert (value - perm_interaction_values[perm_coalition]) < 10e-7


def test_warning_cii():
    """Checks weather a warning is raised for the CHII index and min_order = 0."""
    n = 5
    soum = SOUM(n, n_basis_games=10)
    exact_computer = ExactComputer(game=soum, n_players=n)
    with pytest.warns(UserWarning):
        exact_computer("CHII", 0)

    # check that warning is not raised for min_order > 0
    exact_computer("CHII", 1)


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("Co-Moebius", 2),
        ("Moebius", 2),
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
    """This test checks that the players with the same attribution get the same value."""
    n = 5
    if order is None:
        order = n

    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0.4, 0.2, -0.1, -0.9, 0.4]
        second_order_coefficients = np.asarray(
            [
                [0, 0.4, 0.1, 0, 0],  # interaction btw 0, 1; 0, 2; 1, 4; 2, 4
                [0, 0, 0, 0, 0.4],
                [0, 0, 0, 0, 0.1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )

        def _interaction(arr: np.ndarray):  # dtype bool
            outer = np.outer(arr, arr)
            interaction_array = second_order_coefficients.copy()
            interaction_array[~outer] = 0
            return np.sum(interaction_array)

        value = np.sum(fist_order_coefficients * x_as_float, axis=1)
        interaction_addition = np.apply_along_axis(_interaction, axis=1, arr=X)
        return value + interaction_addition

    exact_computer = ExactComputer(game=_game_fun, n_players=n)
    interaction_values = exact_computer(index=index, order=order)

    # symmetry of players with same attribution
    for coalition in powerset(range(n - 2)):
        coalition_with_first = (0, *tuple([player + 1 for player in coalition]))
        coalition_with_last = (*tuple([player + 1 for player in coalition]), 4)
        assert (
            interaction_values[coalition_with_first] - interaction_values[coalition_with_last]
        ) < 10e-7


@pytest.mark.parametrize(
    ("index", "order"),
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
        ("SII", None),
    ],
)
def test_null_player(index, order):
    """This test checks that the null players don't get any attribution in the values."""
    n = 5
    if order is None:
        order = n

    # game with 0, 4 as null players, has interactions
    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0, 0.2, -0.1, -0.9, 0]
        second_order_coefficients = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0.3, 0],
                [0, 0, 0, 0.4, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )

        def _interaction(arr: np.ndarray):  # dtype bool
            outer = np.outer(arr, arr)
            interaction_array = second_order_coefficients.copy()
            interaction_array[~outer] = 0
            return np.sum(interaction_array)

        value = np.sum(fist_order_coefficients * x_as_float, axis=1)
        interaction_addition = np.apply_along_axis(_interaction, axis=1, arr=X)
        return value + interaction_addition

    exact_computer = ExactComputer(game=_game_fun, n_players=n)
    interaction_values = exact_computer(index=index, order=order)

    # no attribution for coalitions which include the null players.
    for coalition in powerset(range(n - 2)):
        coalition_with_first = (0, *tuple([player + 1 for player in coalition]))
        coalition_with_last = (*tuple([player + 1 for player in coalition]), 4)
        assert interaction_values[coalition_with_first] < 10e-7
        assert interaction_values[coalition_with_last] < 10e-7


@pytest.mark.parametrize(
    ("index", "order"),
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
        ("SII", None),
    ],
)
def test_no_artefact_interaction(index, order):
    """This test checks that the interactions are zero for the game without interactions."""
    n = 5
    if order is None:
        order = n

    # game without interactions
    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0, 0.2, -0.1, -0.9, 0]
        return np.sum(fist_order_coefficients * x_as_float, axis=1)

    exact_computer = ExactComputer(game=_game_fun, n_players=n)
    interaction_values = exact_computer(index=index, order=order)

    for coalition, value in interaction_values.dict_values.items():
        if len(coalition) > 1:
            assert value < 10e-7


@pytest.mark.parametrize(
    ("index", "order"),
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
    """This test checks that the null players don't get any attribution in the generalized values."""
    # implicit in above test for the rest of the indices
    n = 5
    if order is None:
        order = n

    # game with 0, 4 as null players, has interactions
    def _game_fun(X: np.ndarray):
        x_as_float = np.zeros_like(X, dtype=float)
        x_as_float[X] = 1
        fist_order_coefficients = [0, 0.2, -0.1, -0.9, 0]
        second_order_coefficients = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0.3, 0],
                [0, 0, 0, 0.4, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )

        def _interaction(arr: np.ndarray):  # dtype bool
            outer = np.outer(arr, arr)
            interaction_array = second_order_coefficients.copy()
            interaction_array[~outer] = 0
            return np.sum(interaction_array)

        value = np.sum(fist_order_coefficients * x_as_float, axis=1)
        interaction_addition = np.apply_along_axis(_interaction, axis=1, arr=X)
        return value + interaction_addition

    exact_computer = ExactComputer(game=_game_fun, n_players=n)
    interaction_values = exact_computer(index=index, order=order)

    # no attribution for coalitions consisting of the null players.
    assert interaction_values[(0, 4)] < 10e-7
    assert interaction_values[(0,)] < 10e-7
    assert interaction_values[(4,)] < 10e-7
