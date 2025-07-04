"""This test module contains all tests for the base game class in the shapiq package."""

from __future__ import annotations

import os

import numpy as np
import pytest

from shapiq.games.base import Game
from shapiq.games.benchmark import DummyGame  # used to test the base class
from shapiq.utils.sets import powerset, transform_coalitions_to_array


def test_call():
    """This test tests the call function of the base game class."""

    class TestGame(Game):
        """A simple test game that implements the value function."""

        def __init__(self, n, **kwargs):
            super().__init__(n_players=n, normalization_value=0, **kwargs)

        def value_function(self, coalition):
            return np.sum(coalition) / self.n_players

    n_players = 6
    test_game = TestGame(
        n=n_players,
        player_names=["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    )

    # assert that player names are correctly stored
    assert test_game.player_name_lookup == {
        "Alice": 0,
        "Bob": 1,
        "Charlie": 2,
        "David": 3,
        "Eve": 4,
        "Frank": 5,
    }

    assert test_game([]) == 0.0

    # test coalition calls with wrong datatype
    with pytest.raises(TypeError):
        assert test_game([(0, 1), "Alice", "Charlie"])
    with pytest.raises(TypeError):
        assert test_game([(0, 1), ("Alice",), ("Bob",)])
    with pytest.raises(TypeError):
        assert test_game(("Alice", 1))

    # test wrong coalition size in call
    with pytest.raises(TypeError):
        assert test_game(np.array([True, False, True])) == 0.0
    with pytest.raises(TypeError):
        assert test_game(np.array([])) == 0.0

    # test wrong method for numpy array values
    with pytest.raises(TypeError):
        assert test_game(np.array([1, 2, 3, 4, 5, 6])) == 0.0

    # test wrong coalition size in shape[1]
    with pytest.raises(TypeError):
        assert test_game(np.array([[True, False, True]])) == 0.0

    # test with empty coalition all call variants
    test_coalition = test_game.empty_coalition
    assert test_game(test_coalition) == 0.0
    assert test_game(()) == 0.0
    assert test_game([()]) == 0.0

    # test with grand coalition all call variants
    test_coalition = test_game.grand_coalition
    assert test_game(test_coalition) == 1.0
    assert test_game(tuple(range(test_game.n_players))) == 1.0
    assert test_game([tuple(range(test_game.n_players))]) == 1.0
    assert test_game(tuple(test_game.player_name_lookup.values())) == 1.0
    assert test_game([tuple(test_game.player_name_lookup.values())]) == 1.0

    # test with single player coalition all call variants
    test_coalition = np.array([True] + [False for _ in range(test_game.n_players - 1)])
    assert test_game(test_coalition) - 1 / 6 < 10e-7
    assert test_game((0,)) - 1 / 6 < 10e-7
    assert test_game([(0,)]) - 1 / 6 < 10e-7
    assert test_game(("Alice",)) - 1 / 6 < 10e-7
    assert test_game([("Alice",)]) - 1 / 6 < 10e-7

    # test string calls with missing player names
    test_game2 = TestGame(n=n_players)
    with pytest.raises(ValueError):
        assert test_game2(("Bob",)) == 0.0
    with pytest.raises(ValueError):
        assert test_game2([("Charlie",)]) == 0.0


def test_precompute():
    """This test tests the precompute function of the base game class."""
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
    with pytest.raises(KeyError):  # test error case where not all values are precomputed
        _ = dummy_game(dummy_game.empty_coalition)

    # test with large number of players and see if it raises a warning
    with pytest.warns(UserWarning):
        n_players_large = 17
        dummy_game_large = DummyGame(n=n_players_large)
        dummy_game_large.precompute()
        assert dummy_game_large.n_values_stored == 2**n_players_large

    # test empty and grand coalition lookup
    dummy_game = DummyGame(n=4, interaction=(0, 1))
    dummy_game.precompute()
    assert dummy_game.empty_coalition_value == dummy_game(dummy_game.empty_coalition)
    assert dummy_game.grand_coalition_value == dummy_game(dummy_game.grand_coalition)
    assert dummy_game[(0, 1)] == dummy_game[(1, 0)] != 0.0
    with pytest.raises(KeyError):
        _ = dummy_game[(0, 9)]  # only 4 players


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

    # Value error if n_players is None and path_to_values is None:
    with pytest.raises(ValueError):
        _ = Game(n=None, interaction=(0, 1))

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


def test_save_and_load_with_normalization():
    """Tests weather games can be saved and loaded with normalization values correctly."""
    normalization_value = 0.25  # not zero
    dummy_game_empty_output = 0.0

    game = DummyGame(n=4, interaction=(0, 1))
    assert not game.normalize  # first not normalized
    game.normalization_value = normalization_value
    assert game.normalization_value == normalization_value
    assert game.normalize  # now normalized
    assert not game.is_normalized  # the game is being normalized but the empty coalition is not 0

    path = "dummy_game.npz"
    game.save_values(path)

    try:  # do the testings in a try except block to clean up the file
        game_loaded = Game(path_to_values=path)
        assert game_loaded.normalize  # should be normalized
        assert game_loaded.normalization_value == normalization_value
        empty_value = game_loaded(game_loaded.empty_coalition)
        # the output should be the same as the original game with normalization (-0.25)
        assert empty_value == dummy_game_empty_output - normalization_value

        # load with normalization set to False
        game_loaded = Game(path_to_values=path, normalize=False)
        assert not game_loaded.normalize  # should not be normalized
        assert game_loaded.normalization_value == 0.0
        empty_value = game_loaded(game_loaded.empty_coalition)
        # the output should be the same as the original game without normalization (0.0)
        assert empty_value == dummy_game_empty_output
    except Exception:
        raise
    finally:
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


def test_abstract_game():
    """Tests the abstract game class."""
    from tests.utils import get_concrete_class

    n = 6
    game = get_concrete_class(Game)(n_players=n)
    with pytest.raises(NotImplementedError):
        game(np.array([[True for _ in range(n)]]))


def test_exact_computer_call():
    """Tests the call to the exact computer in the game class."""
    game = DummyGame(n=4, interaction=(0, 1))

    index = "SII"
    order = 2
    sv = game.exact_values(index=index, order=order)
    assert sv.index == index
    assert sv.max_order == order


def test_compute():
    """Tests the compute function with and without returned normalization."""
    normalization_value = 1.0  # not zero

    n_players = 3
    game = DummyGame(n=n_players, interaction=(0, 1))

    coalitions = np.array([[1, 0, 0], [0, 1, 1]])

    # Make sure normalization value is added
    game.normalization_value = normalization_value
    assert game.normalize

    result = game.compute(coalitions=coalitions)
    assert len(result[0]) == len(coalitions)  # number of coalitions is correct
    assert result[2] == normalization_value
    assert len(result) == 3  # game_values, normalization_value and coalition_lookup

    # check if the game values are correct and that they are not normalized from compute
    game_values = result[0]
    assert game(coalitions[0]) + normalization_value == pytest.approx(game_values[0])
    assert game(coalitions[1]) + normalization_value == pytest.approx(game_values[1])
