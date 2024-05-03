"""This test module contains all tests for the base game class in the shapiq package."""

import os

import numpy as np
import pytest


from shapiq.games.benchmark import DummyGame  # used to test the base class
from shapiq.utils.sets import powerset, transform_coalitions_to_array


def test_precompute():
    """This test tests the precompute function of the base game class"""
    n_players = 6
    dummy_game = DummyGame(n=n_players, interaction=(0, 1))

    assert dummy_game.n_values_stored == 0
    assert len(dummy_game.value_storage) == 0  # no precomputed values
    assert len(dummy_game.coalition_lookup) == 0  # empty base attribute
    assert dummy_game.n_players == n_players  # base attribute

    dummy_game.precompute()

    assert len(dummy_game.value_storage) != 0  # precomputed values
    assert len(dummy_game.coalition_lookup) != 0  # precomputed coalitions
    assert dummy_game.value_storage.shape[0] == 2**n_players  # precomputed values
    assert dummy_game.n_values_stored == 2**n_players  # precomputed coalitions
    assert dummy_game.precomputed  # precomputed flag

    # test with coalitions param provided
    n_players = 6
    dummy_game = DummyGame(n=n_players, interaction=(0, 1))
    coalitions = np.array([[True for _ in range(n_players)]])
    dummy_game.precompute(coalitions=coalitions)

    assert dummy_game.n_values_stored == 1

    with pytest.warns(UserWarning):
        n_players_large = 17
        dummy_game_large = DummyGame(n=n_players_large)
        # call precompute but stop it before it finishes
        dummy_game_large.precompute()

        assert dummy_game_large.n_values_stored == 2**n_players_large


def test_core_functions():
    """This test tests the core functions of the base game class object."""

    n_players = 6
    dummy_game = DummyGame(n=n_players, interaction=(0, 1))

    # test repr and str
    string_game = str(dummy_game)
    assert isinstance(repr(dummy_game), str)
    assert isinstance(str(dummy_game), str)
    assert repr(dummy_game) == string_game
    assert dummy_game.game_name == "DummyGame_Game"

    dummy_game.normalization_value = 1.0
    assert dummy_game.normalize  # should be true if normalization_value is not 0.0


def test_lookup_vs_run():
    """Tests weather games are correctly evaluated by the lookup table or the value function."""
    dummy_game_precomputed = DummyGame(n=4, interaction=(0, 1))
    dummy_game_precomputed.precompute()

    dummy_game = DummyGame(n=4, interaction=(0, 1))

    test_coalition = dummy_game.empty_coalition
    assert np.allclose(dummy_game(test_coalition), dummy_game_precomputed(test_coalition))

    test_coalition = dummy_game.empty_coalition
    test_coalition[0] = True
    assert np.allclose(dummy_game(test_coalition), dummy_game_precomputed(test_coalition))

    test_coalition = dummy_game.grand_coalition
    assert np.allclose(dummy_game(test_coalition), dummy_game_precomputed(test_coalition))


def test_load_and_save():
    """This test tests the save and load functions of the base game class object."""

    dummy_game = DummyGame(n=4, interaction=(0, 1))
    dummy_game.precompute()
    path = "dummy_game.pkl"
    dummy_game.save(path)

    assert os.path.exists(path)

    dummy_game_loaded = DummyGame.load(path)

    assert dummy_game.value_storage.shape == dummy_game_loaded.value_storage.shape
    assert len(dummy_game.coalition_lookup) == len(dummy_game_loaded.coalition_lookup)
    assert dummy_game.n_values_stored == dummy_game_loaded.n_values_stored
    assert dummy_game.n_players == dummy_game_loaded.n_players
    assert dummy_game.interaction == dummy_game_loaded.interaction
    assert np.all(dummy_game.value_storage == dummy_game_loaded.value_storage)
    # check if dict is the same
    assert dummy_game.coalition_lookup == dummy_game_loaded.coalition_lookup
    assert dummy_game.normalize == dummy_game_loaded.normalize
    assert dummy_game.precomputed == dummy_game_loaded.precomputed

    # clean up
    os.remove(path)
    assert not os.path.exists(path)

    # test store values and load

    path = "dummy_game.npz"
    dummy_game.save_values(path)

    assert os.path.exists(path)

    dummy_game_loaded = DummyGame(n=4, interaction=(0, 1))
    dummy_game_loaded.load_values(path)

    assert dummy_game.value_storage.shape == dummy_game_loaded.value_storage.shape
    assert len(dummy_game.coalition_lookup) == len(dummy_game_loaded.coalition_lookup)
    assert dummy_game.n_players == dummy_game_loaded.n_players
    assert dummy_game.interaction == dummy_game_loaded.interaction
    assert np.allclose(dummy_game.value_storage, dummy_game_loaded.value_storage)
    assert dummy_game.precomputed == dummy_game_loaded.precomputed

    # clean up
    os.remove(path)
    assert not os.path.exists(path)

    # path without .npz
    path = "dummy_game"
    dummy_game.save_values(path)

    assert os.path.exists(path + ".npz")

    # load without .npz as path ending
    dummy_game_loaded = DummyGame(n=4, interaction=(0, 1))
    dummy_game_loaded.load_values(path)
    assert dummy_game.n_values_stored == dummy_game_loaded.n_values_stored

    # load with wrong number of players expect ValueError
    dummy_game_loaded = DummyGame(n=5, interaction=(0, 1))
    with pytest.raises(ValueError):
        dummy_game_loaded.load_values(path + ".npz")

    # clean up
    os.remove(path + ".npz")
    assert not os.path.exists(path + ".npz")

    # test without precomputed values and see if it raises a warning
    path = "dummy_game.npz"
    dummy_game = DummyGame(n=4, interaction=(0, 1))
    with pytest.warns(UserWarning):
        dummy_game.save_values(path)

    assert os.path.exists(path)

    # clean up
    os.remove(path)
    assert not os.path.exists(path)


def test_progress_bar():
    """Tests the progress bar of the game class."""

    dummy_game = DummyGame(n=5, interaction=(0, 1))

    test_coalitions = list(powerset(range(dummy_game.n_players)))
    test_coalitions = transform_coalitions_to_array(test_coalitions, dummy_game.n_players)

    # check if the progress bar is displayed

    values = dummy_game(test_coalitions, verbose=True)
    assert len(values) == len(test_coalitions)
