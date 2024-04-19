"""This module contains the tests for the unsupervised data benchmark games."""

import numpy as np

from shapiq.games.base import Game
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
    game = UnsupervisedData(
        data=data,
        normalize=True,
    )
    assert isinstance(game, Game)
    assert isinstance(game, UnsupervisedData)
    assert game.n_players == n_players

    # test value function
    coalitions = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]).astype(bool)
    values = game(coalitions)
    assert values.shape == (4,)
    assert values[0] == 0.0  # should be zero
    assert np.all(values[1:] != 0.0)  # rest should not be zero


def test_adult():
    """This function tests the adult census unsupervised data game."""
    n_players = 14
    # setup game
    game = AdultCensusUnsupervisedData(normalize=True)
    assert isinstance(game, Game)
    assert isinstance(game, UnsupervisedData)
    assert game.n_players == n_players
    # no test for value function as it takes too long


def test_bike_sharing():
    """This function tests the bike sharing unsupervised data game."""
    n_players = 12
    # setup game
    game = BikeSharingUnsupervisedData(normalize=True)
    assert isinstance(game, Game)
    assert isinstance(game, UnsupervisedData)
    assert game.n_players == n_players
    # no test for value function as it takes too long


def test_california_housing():
    """This function tests the california housing unsupervised data game."""
    n_players = 8
    # setup game
    game = CaliforniaHousingUnsupervisedData(normalize=True)
    assert isinstance(game, Game)
    assert isinstance(game, UnsupervisedData)
    assert game.n_players == n_players
    # no test for value function as it takes too long
