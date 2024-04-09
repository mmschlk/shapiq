"""This test module contains all tests regarding the AdultCensus."""

import os

import numpy as np
import pytest

from shapiq.games import AdultCensus


@pytest.mark.slow
@pytest.mark.parametrize("model", ["sklearn_rf", "invalid"])
def test_basic_function(model):
    """Tests the AdultCensus game."""

    game_n_players = 14

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = AdultCensus(model=model, x_explain=0)
        return

    x_explain_id = 1
    game = AdultCensus(x_explain=x_explain_id, model=model)
    assert game.n_players == game_n_players

    # test full prediction output against underlying model
    full_pred = float(game(np.ones(game_n_players, dtype=bool)))
    assert full_pred + game.normalization_value == 0.28  # for x_explain_id=1 it should be 0.28

    test_coalitions_precompute = np.array(
        [
            np.zeros(game_n_players),
            np.ones(game_n_players),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]),
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
    new_game = AdultCensus(path_to_values=path)
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert np.allclose(new_game.value_storage, game.value_storage)

    # clean up
    os.remove(path)
    assert not os.path.exists(path)

    # value error for wrong class
    with pytest.raises(ValueError):
        _ = AdultCensus(x_explain=x_explain_id, class_to_explain=2)
