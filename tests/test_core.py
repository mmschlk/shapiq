"""This test module tests the core calculations"""

import numpy as np
import pytest

import shapiq
from shapiq.core import egalitarian_least_core
from shapiq.games.benchmark.synthetic.soum import SOUM
from shapiq.utils import powerset


def test_core_on_soum():
    for _ in range(20):
        n = np.random.randint(low=2, high=10)
        n_basis_games = np.random.randint(low=1, high=100)
        soum = SOUM(n, n_basis_games=n_basis_games)

        coalition_lookup = {}
        coalition_matrix = np.zeros((2**n, n), dtype=bool)
        grand_coalition_set = set(range(n))
        for i, T in enumerate(powerset(grand_coalition_set, min_size=0, max_size=n)):
            coalition_lookup[T] = i  # set lookup for the coalition
            coalition_matrix[i, T] = True  # one-hot-encode the coalition
        game_values = soum(coalition_matrix)  # compute the game values
        baseline_value = float(game_values[0])  # set the baseline value

        predicted_value = soum(np.ones(n))[0]  # value of grand coalition

        egalitarian_vector, subsidy = egalitarian_least_core(
            n_players=n, game_values=game_values, coalition_lookup=coalition_lookup
        )

        # Assert efficiency
        assert (np.sum(egalitarian_vector.values) + baseline_value - predicted_value) ** 2 < 10e-7

        stability_equations = coalition_matrix[:-1] @ egalitarian_vector.values + subsidy
        game_values = game_values[:-1] - baseline_value

        # Assert stability
        assert np.all(stability_equations - game_values >= -10e-7)


def test_core_on_normalized_soum():
    for _ in range(20):
        n = np.random.randint(low=2, high=10)
        n_basis_games = np.random.randint(low=1, high=100)
        soum = SOUM(n, n_basis_games=n_basis_games, normalize=True)

        coalition_lookup = {}
        coalition_matrix = np.zeros((2**n, n), dtype=bool)
        grand_coalition_set = set(range(n))
        for i, T in enumerate(powerset(grand_coalition_set, min_size=0, max_size=n)):
            coalition_lookup[T] = i
            coalition_matrix[i, T] = True  # one-hot-encode the coalition
        game_values = soum(coalition_matrix)  # compute the game values
        predicted_value = soum(np.ones(n))[0]  # value of grand coalition

        egalitarian_vector, subsidy = egalitarian_least_core(
            n_players=n, game_values=game_values, coalition_lookup=coalition_lookup
        )

        # Assert efficiency.
        assert (np.sum(egalitarian_vector.values) - predicted_value) ** 2 < 10e-7

        stability_equations = coalition_matrix[:-1] @ egalitarian_vector.values + subsidy
        game_values = game_values[:-1]

        # Assert stability
        assert np.all(stability_equations - game_values >= -10e-7)


def test_core_political_game_empty_core():
    """Tests that core is empty for non-convex game and egalitarian least-core has subsidy=33.3.

    The political game tested here is constructed such that the core is [33.3, 33.3, 33.3] with subsidy 33.3.
    This is due to all coalitions with at least two players gets 100.

    """

    class NonConvexGame(shapiq.Game):

        def __init__(self) -> None:
            super().__init__(n_players=3, normalize=True, normalization_value=0)

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            coalition_values = {
                (): 0,
                (0,): 0,
                (1,): 0,
                (2,): 0,
                (0, 1): 100,
                (0, 2): 100,
                (1, 2): 100,
                (0, 1, 2): 100,
            }

            values = np.array([coalition_values[tuple(np.where(x)[0])] for x in coalitions])

            return values

    game_political = NonConvexGame()
    coalition_lookup = {}
    coalition_matrix = np.zeros((2**game_political.n_players, game_political.n_players), dtype=bool)
    grand_coalition_set = set(range(3))
    for i, T in enumerate(
        powerset(grand_coalition_set, min_size=0, max_size=game_political.n_players)
    ):
        coalition_lookup[T] = i  # set lookup for the coalition
        coalition_matrix[i, T] = True  # one-hot-encode the coalition
    game_values = game_political(coalition_matrix)  # compute the game values
    baseline_value = float(game_values[0])  # set the baseline value

    predicted_value = game_political(np.ones(3))[0]  # value of grand coalition

    egalitarian_vector, subsidy = egalitarian_least_core(
        n_players=3, game_values=game_values, coalition_lookup=coalition_lookup
    )
    # Assert correct values
    assert np.all(egalitarian_vector.values - np.array([100 / 3, 100 / 3, 100 / 3]) < 10e-7)
    assert subsidy - 100 / 3 < 10e-7

    # Assert efficiency
    assert (np.sum(egalitarian_vector.values) + baseline_value - predicted_value) ** 2 < 10e-7


def test_core_baseline_warning():
    game_values = np.array([10, 5, 20, 100])

    with pytest.warns(UserWarning):
        egalitarian_least_core(
            n_players=2,
            game_values=game_values,
            coalition_lookup={tuple(): 0, (0,): 1, (1,): 2, (0, 1): 3},
        )


def test_core_political_game_existing_core():
    """Tests that the ELC is equal to the core with subsidy equal to 0, due to convex game structure."""

    class ConvexGame(shapiq.Game):
        """Convex game, i.e. meaning that the v(S u {i}) - v(S) <= v(T u {i}) - v(T) for S<=T<={1,..,n} \ {i}.
        The marginal contribution of a player i is always bigger if it joins a bigger coalition.
        """

        def __init__(self) -> None:
            super().__init__(n_players=3, normalize=True, normalization_value=0)

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            coalition_values = {
                (): 0,
                (0,): 0,
                (1,): 0,
                (2,): 0,
                (0, 1): 100,
                (0, 2): 100,
                (1, 2): 100,
                (0, 1, 2): 200,
            }

            values = np.array([coalition_values[tuple(np.where(x)[0])] for x in coalitions])

            return values

    game_political = ConvexGame()
    coalition_lookup = {}
    coalition_matrix = np.zeros((2**game_political.n_players, game_political.n_players), dtype=bool)
    grand_coalition_set = set(range(3))
    for i, T in enumerate(
        powerset(grand_coalition_set, min_size=0, max_size=game_political.n_players)
    ):
        coalition_lookup[T] = i  # set lookup for the coalition
        coalition_matrix[i, T] = True  # one-hot-encode the coalition
    game_values = game_political(coalition_matrix)  # compute the game values
    baseline_value = float(game_values[0])  # set the baseline value

    predicted_value = game_political(np.ones(3))[0]  # value of grand coalition

    egalitarian_vector, subsidy = egalitarian_least_core(
        n_players=3, game_values=game_values, coalition_lookup=coalition_lookup
    )
    # Assert correct values
    assert np.all(egalitarian_vector.values - np.array([200 / 3, 200 / 3, 200 / 3]) < 10e-7)
    pytest.approx(subsidy, rel=0, abs=1e-5)

    # Assert efficiency
    assert (np.sum(egalitarian_vector.values) + baseline_value - predicted_value) ** 2 < 10e-7
