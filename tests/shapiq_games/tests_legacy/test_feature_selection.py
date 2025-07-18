"""This test module contains all tests regarding the FeatureSelection game."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from shapiq.games.benchmark import (
    AdultCensusFeatureSelection,
    BikeSharingFeatureSelection,
    CaliforniaHousingFeatureSelection,
    FeatureSelection,
)


def _loss_function(y_pred, y_test):
    """A simple loss function for regression tasks."""
    return np.mean(np.square(y_pred - y_test))


def test_basic_function(background_reg_dataset):
    """Tests the FeatureSelection game with a small regression dataset."""
    x_data, y_data = background_reg_dataset
    x_data = x_data[:, :3]  # get first three features

    model = DecisionTreeRegressor(max_depth=4)

    # init game with score function
    game = FeatureSelection(
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

    game = FeatureSelection(
        x_train=x_data,
        x_test=x_data,
        y_train=y_data,
        y_test=y_data,
        fit_function=model.fit,
        predict_function=model.predict,
        loss_function=_loss_function,
    )

    game.precompute()

    assert game.n_values_stored == 2**3

    # init with no score and or predict function
    with pytest.raises(ValueError):
        _ = FeatureSelection(
            x_train=x_data,
            x_test=x_data,
            y_train=y_data,
            y_test=y_data,
            fit_function=model.fit,
        )


@pytest.mark.parametrize("model_name", ["decision_tree", "random_forest", "gradient_boosting"])
def test_california(model_name):
    """Test the FeatureSelection game with the california housing dataset."""
    n_players = 8
    test_coalition = np.zeros(shape=(1, n_players), dtype=bool)
    test_coalition[0][0] = True

    game = CaliforniaHousingFeatureSelection(model_name=model_name)
    value = game(test_coalition)
    assert game.n_players == n_players
    assert len(value) == 1
    assert game.game_name == "CaliforniaHousing_FeatureSelection_Game"


@pytest.mark.parametrize("model_name", ["decision_tree", "random_forest", "gradient_boosting"])
def test_adult_census(model_name):
    """Test the FeatureSelection game with the adult census dataset."""
    n_players = 14
    test_coalition = np.zeros(shape=(1, n_players), dtype=bool)
    test_coalition[0][0] = True

    game = AdultCensusFeatureSelection(model_name=model_name)
    value = game(test_coalition)
    assert game.n_players == n_players
    assert len(value) == 1
    assert game.game_name == "AdultCensus_FeatureSelection_Game"


@pytest.mark.parametrize("model_name", ["decision_tree", "random_forest", "gradient_boosting"])
def test_bike_sharing(model_name):
    """Test the FeatureSelection game with the bike sharing dataset."""
    n_players = 12
    test_coalition = np.zeros(shape=(1, n_players), dtype=bool)
    test_coalition[0][0] = True

    game = BikeSharingFeatureSelection(model_name=model_name)
    value = game(test_coalition)
    assert game.n_players == n_players
    assert len(value) == 1
    assert game.game_name == "BikeSharing_FeatureSelection_Game"
