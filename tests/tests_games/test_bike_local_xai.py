"""This test module contains all tests regarding the BikeRegression."""

import os

import numpy as np
import pytest

from shapiq.games import BikeRegression


@pytest.mark.parametrize("model", ["xgboost", "invalid"])
def test_basic_function(model):
    """Tests the BikeRegression game."""

    game_n_players = 12

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = BikeRegression(model=model, x_explain=0)
        return

    x_explain_id = 0
    game = BikeRegression(x_explain=x_explain_id, model=model)
    assert game.n_players == game_n_players

    test_coalitions_precompute = np.array(
        [
            np.zeros(game_n_players),
            np.ones(game_n_players),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        ],
        dtype=bool,
    )

    game.precompute(coalitions=test_coalitions_precompute)
    assert game.n_players == game_n_players
    assert len(game.feature_names) == game_n_players
    assert game.n_values_stored == 4
    assert game.precomputed

    # test save and load values
    path = f"test_values_bike.npz"
    game.save_values(path)
    assert os.path.exists(path)

    # test init from values file
    new_game = BikeRegression(path_to_values=path)
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert np.allclose(new_game.value_storage, game.value_storage)

    # clean up
    os.remove(path)
    assert not os.path.exists(path)
