"""This test module contains all tests regarding the FeatureSelection game."""
import os

import numpy as np

from shapiq.games.tabular import LocalExplanation


def test_basic_function(background_reg_dataset, dt_reg_model):
    """Tests the FeatureSelection game with a small regression dataset."""
    x_data, y_data = background_reg_dataset
    x_explain = x_data[0].copy()  # get first instance

    model = dt_reg_model

    # init game
    game = LocalExplanation(x_data=x_data, model=model.predict, x_explain=x_explain)
    game.precompute()

    # test save and load
    game.save("test_game.pkl")
    assert os.path.exists("test_game.pkl")

    # load new game
    new_game = LocalExplanation.load("test_game.pkl")
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert new_game.precomputed == game.precomputed

    # compare output on same input
    test_coalition = new_game.empty_coalition
    test_coalition[0] = True
    assert new_game(test_coalition) == game(test_coalition)

    # clean up
    os.remove("test_game.pkl")
    assert not os.path.exists("test_game.pkl")

    # init game with integer
    game = LocalExplanation(x_data=x_data, model=model.predict, x_explain=0)
    # check if the x_explain is valid
    assert np.all(game.x_explain == x_data[0])
