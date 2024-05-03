"""This module contains the tests for the unsupervised data benchmark games."""

import numpy as np

from shapiq.games.base import Game
from shapiq.utils import powerset
from shapiq.games.benchmark import UnsupervisedData
from shapiq.games.benchmark import (
    AdultCensusUnsupervisedData,
    BikeSharingUnsupervisedData,
    CaliforniaHousingUnsupervisedData,
)


def test_base_class():
    """This function tests the setup and logic of the game."""

    n_players = 4

    # create synthetic data
    data = np.random.rand(200, n_players)

    # setup game
    game = UnsupervisedData(data=data)
    assert isinstance(game, Game)
    assert isinstance(game, UnsupervisedData)
    assert game.n_players == n_players

    # test value function on all coalitions
    coalitions = np.zeros((2**n_players, n_players), dtype=bool)
    for i, coalition in enumerate(powerset(range(n_players))):
        coalitions[i, list(coalition)] = True
    values = game(coalitions)

    # check if the values are correct
    assert values.shape == (2**n_players,)

    for i, coalition in enumerate(powerset(range(n_players))):
        if len(coalition) <= 1:
            assert values[i] == 0.0  # must be zero for empty and single player coalitions
        else:
            assert values[i] != 0.0  # should be non-zero for non-empty coalitions


def test_adult():
    """This function tests the adult census unsupervised data game."""
    n_players = 14
    # setup game
    game = AdultCensusUnsupervisedData()
    assert isinstance(game, Game)
    assert isinstance(game, UnsupervisedData)
    assert game.n_players == n_players
    assert game.game_name == "AdultCensus_UnsupervisedData_Game"

    test_coalitions = np.array(
        [game.empty_coalition, game.empty_coalition, game.grand_coalition]
    ).astype(bool)
    test_coalitions[1][2] = True  # one player coalition

    test_values = game(test_coalitions)
    assert test_values.shape == (3,)
    assert test_values[0] == 0.0
    assert test_values[1] == 0.0
    assert test_values[2] != 0.0


def test_bike_sharing():
    """This function tests the bike sharing unsupervised data game."""
    n_players = 12
    # setup game
    game = BikeSharingUnsupervisedData()
    assert isinstance(game, Game)
    assert isinstance(game, UnsupervisedData)
    assert game.n_players == n_players
    assert game.game_name == "BikeSharing_UnsupervisedData_Game"

    test_coalitions = np.array(
        [game.empty_coalition, game.empty_coalition, game.grand_coalition]
    ).astype(bool)
    test_coalitions[1][2] = True  # one player coalition

    test_values = game(test_coalitions)
    assert test_values.shape == (3,)
    assert test_values[0] == 0.0
    assert test_values[1] == 0.0
    assert test_values[2] != 0.0


def test_california_housing():
    """This function tests the california housing unsupervised data game."""
    n_players = 8
    # setup game
    game = CaliforniaHousingUnsupervisedData()
    assert isinstance(game, Game)
    assert isinstance(game, UnsupervisedData)
    assert game.n_players == n_players
    assert game.game_name == "CaliforniaHousing_UnsupervisedData_Game"

    test_coalitions = np.array(
        [game.empty_coalition, game.empty_coalition, game.grand_coalition]
    ).astype(bool)
    test_coalitions[1][2] = True  # one player coalition

    test_values = game(test_coalitions)
    assert test_values.shape == (3,)
    assert test_values[0] == 0.0
    assert test_values[1] == 0.0
    assert test_values[2] != 0.0
