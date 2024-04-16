"""This test module contains all tests regarding the FeatureSelection game."""

import os

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from shapiq.games.feature_selection import FeatureSelectionGame


def loss_function(y_pred, y_test):
    return np.mean(np.square(y_pred - y_test))


def test_basic_function(background_reg_dataset):
    """Tests the FeatureSelection game with a small regression dataset."""
    x_data, y_data = background_reg_dataset
    x_data = x_data[:, :3]  # get first three features

    model = DecisionTreeRegressor(max_depth=4)

    # init game with score function
    game = FeatureSelectionGame(
        x_train=x_data,
        x_test=x_data,
        y_train=y_data,
        y_test=y_data,
        fit_function=model.fit,
        score_function=model.score,
    )
    game.precompute()

    assert game.n_values_stored == 2**3

    # init game with predict and loss function

    game = FeatureSelectionGame(
        x_train=x_data,
        x_test=x_data,
        y_train=y_data,
        y_test=y_data,
        fit_function=model.fit,
        predict_function=model.predict,
        loss_function=loss_function,
    )

    game.precompute()

    assert game.n_values_stored == 2**3

    # init with no score and or predict function
    with pytest.raises(ValueError):
        _ = FeatureSelectionGame(
            x_train=x_data, x_test=x_data, y_train=y_data, y_test=y_data, fit_function=model.fit
        )

    # test save and load
    game.save("test_game.pkl")
    assert os.path.exists("test_game.pkl")

    # load new game
    new_game = FeatureSelectionGame.load("test_game.pkl")
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert new_game.precomputed == game.precomputed

    # clean up
    os.remove("test_game.pkl")
    assert not os.path.exists("test_game.pkl")

    # test with path_to_values
    game.save_values("test_values.npz")
    new_game = FeatureSelectionGame(path_to_values="test_values.npz")
    assert new_game.n_values_stored == game.n_values_stored

    # clean up
    os.remove("test_values.npz")
    assert not os.path.exists("test_values.npz")


def test_california():
    """Test the FeatureSelection game with the california housing dataset."""
    raise NotImplementedError("The game is not implemented yet.")


def test_adult_census():
    """Test the FeatureSelection game with the adult census dataset."""
    raise NotImplementedError("The game is not implemented yet.")


def test_bike_sharing():
    """Test the FeatureSelection game with the bike sharing dataset."""
    raise NotImplementedError("The game is not implemented yet.")
