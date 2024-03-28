"""This test module contains all tests for the base game class in the shapiq package."""

import os

import numpy as np
import pytest


from shapiq.games.base import Game
from shapiq.games.dummy import DummyGame  # used to test the base class


def test_precompute():
    """This test tests the precompute function of the base game class"""
    n_players = 6
    dummy_game = DummyGame(n=n_players, interaction=(0, 1))

    assert dummy_game.value_storage is None  # empty base attribute
    assert dummy_game.coalitions_in_storage is None  # empty base attribute
    assert dummy_game.n_players == n_players  # base attribute
    assert dummy_game.empty_value == 0.0  # base attribute (zero for dummy game)

    dummy_game.precompute()

    assert dummy_game.value_storage is not None  # precomputed values
    assert dummy_game.coalitions_in_storage is not None  # precomputed coalitions
    assert dummy_game.value_storage.shape[0] == 2**n_players  # precomputed values
    assert dummy_game.value_storage[0] == dummy_game.empty_value  # empty coalition value

    # test with coalitions param provided
    coalitions = np.array([[True for _ in range(n_players)]])
    dummy_game.precompute(coalitions=coalitions)

    assert len(dummy_game.value_storage) == len(coalitions)  # only the ones specified were run
    assert len(dummy_game.coalitions_in_storage) == len(coalitions)  # only the ones specified

    with pytest.warns(UserWarning):
        n_players_large = 17
        dummy_game_large = DummyGame(n=n_players_large)
        # call precompute but stop it before it finishes
        dummy_game_large.precompute()

        assert len(dummy_game_large.value_storage) == 2**n_players_large
        assert len(dummy_game_large.coalitions_in_storage) == 2**n_players_large


def test_core_functions():
    """This test tests the core functions of the base game class object."""

    n_players = 6
    dummy_game = DummyGame(n=n_players, interaction=(0, 1))

    # test repr and str
    string_game = str(dummy_game)
    assert isinstance(repr(dummy_game), str)
    assert isinstance(str(dummy_game), str)
    assert repr(dummy_game) == string_game


def test_load_and_save():
    """This test tests the save and load functions of the base game class object."""

    dummy_game = DummyGame(n=4, interaction=(0, 1))
    dummy_game.precompute()
    path = "dummy_game.pkl"
    dummy_game.save(path)

    assert os.path.exists(path)

    dummy_game_loaded = DummyGame.load(path)

    assert dummy_game.value_storage.shape == dummy_game_loaded.value_storage.shape
    assert dummy_game.coalitions_in_storage.shape == dummy_game_loaded.coalitions_in_storage.shape
    assert dummy_game.empty_value == dummy_game_loaded.empty_value
    assert dummy_game.n_players == dummy_game_loaded.n_players
    assert dummy_game.interaction == dummy_game_loaded.interaction
    assert np.all(dummy_game.value_storage == dummy_game_loaded.value_storage)
    assert np.all(dummy_game.coalitions_in_storage == dummy_game_loaded.coalitions_in_storage)

    # clean up
    os.remove(path)

    # test store values and load

    path = "dummy_game.npz"
    dummy_game.save_values(path)

    assert os.path.exists(path)

    dummy_game_loaded = DummyGame(n=4, interaction=(0, 1))
    dummy_game_loaded.load_values(path)

    assert dummy_game.value_storage.shape == dummy_game_loaded.value_storage.shape
    assert dummy_game.coalitions_in_storage.shape == dummy_game_loaded.coalitions_in_storage.shape
    assert dummy_game.empty_value == dummy_game_loaded.empty_value
    assert dummy_game.n_players == dummy_game_loaded.n_players
    assert dummy_game.interaction == dummy_game_loaded.interaction
    assert np.allclose(dummy_game.value_storage, dummy_game_loaded.value_storage)
    assert np.all(dummy_game.coalitions_in_storage == dummy_game_loaded.coalitions_in_storage)

    # clean up
    os.remove(path)

    # path without .npz
    path = "dummy_game"
    dummy_game.save_values(path)

    assert os.path.exists(path + ".npz")

    # load without .npz as path ending
    dummy_game_loaded = DummyGame(n=4, interaction=(0, 1))
    dummy_game_loaded.load_values(path)

    # load with wrong number of players expect ValueError
    dummy_game_loaded = DummyGame(n=5, interaction=(0, 1))
    with pytest.raises(ValueError):
        dummy_game_loaded.load_values(path + ".npz")

    # clean up
    os.remove(path + ".npz")

    # test without precomputed values and see if it raises a warning
    path = "dummy_game.npz"
    dummy_game = DummyGame(n=4, interaction=(0, 1))
    with pytest.warns(UserWarning):
        dummy_game.save_values(path)

    assert os.path.exists(path)

    # clean up
    os.remove(path)
