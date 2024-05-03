"""This test module contains all tests regarding the GlobalExplanation game."""

import os

import numpy as np
import pytest

from shapiq.games import Game
from shapiq.games.benchmark import GlobalExplanation
from shapiq.games.benchmark import (
    AdultCensusGlobalXAI,
    BikeSharingGlobalXAI,
    CaliforniaHousingGlobalXAI,
)


def test_basic_function(background_reg_dataset, dt_reg_model, mae_loss):
    """Tests the GlobalExplanation game with a small regression and classification dataset."""
    data, target = background_reg_dataset
    n_players = data.shape[1]
    test_coalitions = np.array(
        [
            [False for _ in range(n_players)],
            [True for _ in range(n_players)],
        ]
    ).astype(bool)

    game = GlobalExplanation(
        data=data,
        model=dt_reg_model.predict,
        loss_function=mae_loss,
        n_samples_empty=10,
        random_state=42,
        normalize=True,
    )
    assert game.n_players == n_players

    # test the value function
    values = game(test_coalitions)
    empty_value = game.normalization_value
    assert values.shape == (2,)
    assert empty_value != 0.0  # empty value is not expected to be zero

    # test pre-compute
    game.precompute(test_coalitions)
    assert game.n_values_stored == 2

    # test save and load from values
    game.save_values("test_values.npz")
    assert os.path.exists("test_values.npz")

    new_game = Game(path_to_values="test_values.npz")
    stored_values = new_game(test_coalitions)
    assert new_game.n_players == n_players
    assert new_game.n_values_stored == 2

    # only all coalitions that are not empty set are expected to be the same re-running the empty
    # set prediction is not guaranteed to be the same which still is an open problem in global xai
    # with feature attributions
    assert np.allclose(values[1], stored_values[1])

    # clean up
    os.remove("test_values.npz")
    assert not os.path.exists("test_values.npz")


@pytest.mark.parametrize("model_name", ["decision_tree", "random_forest", "gradient_boosting"])
def test_california(model_name):
    """Test the GlobalExplanation game with the california housing dataset."""
    test_coalitions = np.zeros(shape=(2, 8), dtype=bool)
    test_coalitions[1] = np.ones(8, dtype=bool)
    game = CaliforniaHousingGlobalXAI(
        model_name=model_name, n_samples_eval=2, n_samples_empty=3  # small values for testing
    )
    worth = game(test_coalitions)
    assert game.n_players == 8
    assert worth.shape == (2,)
    assert game.game_name == "CaliforniaHousing_GlobalExplanation_Game"


@pytest.mark.parametrize("model_name", ["decision_tree", "random_forest", "gradient_boosting"])
def test_adult_census(model_name):
    """Test the GlobalExplanation game with the adult census dataset."""
    test_coalitions = np.zeros(shape=(2, 14), dtype=bool)
    test_coalitions[1] = np.ones(14, dtype=bool)
    game = AdultCensusGlobalXAI(
        model_name=model_name, n_samples_eval=2, n_samples_empty=3  # small values for testing
    )
    worth = game(test_coalitions)
    assert game.n_players == 14
    assert worth.shape == (2,)
    assert game.game_name == "AdultCensus_GlobalExplanation_Game"


@pytest.mark.parametrize("model_name", ["decision_tree", "random_forest", "gradient_boosting"])
def test_bike_sharing(model_name):
    """Test the GlobalExplanation game with the bike sharing dataset."""
    test_coalitions = np.zeros(shape=(2, 12), dtype=bool)
    test_coalitions[1] = np.ones(12, dtype=bool)
    game = BikeSharingGlobalXAI(
        model_name=model_name, n_samples_eval=2, n_samples_empty=3  # small values for testing
    )
    worth = game(test_coalitions)
    assert game.n_players == 12
    assert worth.shape == (2,)
    assert game.game_name == "BikeSharing_GlobalExplanation_Game"
