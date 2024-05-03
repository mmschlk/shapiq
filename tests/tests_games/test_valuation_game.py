"""This module contains the tests for the `DatasetValuationGame` class."""

import os

import numpy as np
import pytest

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

from shapiq.games.base import Game
from shapiq.games.benchmark import (
    DatasetValuation,
    AdultCensusDatasetValuation,
    BikeSharingDatasetValuation,
    CaliforniaHousingDatasetValuation,
)


def test_dataset_valuation_game(background_reg_dataset, background_clf_dataset):
    """This function tests the setup and logic of the game."""

    n_players = 4
    player_sizes = [0.25, 0.25, 0.4, 0.1]
    test_coalitions = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]]
    test_coalitions = np.asarray(test_coalitions).astype(bool)

    # test with regression dataset
    x_train, y_train = background_reg_dataset
    x_test, y_test = background_reg_dataset
    model = DecisionTreeRegressor()

    # setup game
    game = DatasetValuation(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        fit_function=model.fit,
        predict_function=model.predict,
        loss_function=mean_squared_error,
        player_sizes=player_sizes,
        random_state=42,
    )
    game_values = game(test_coalitions)
    assert game.n_players == n_players
    assert len(game.data_sets) == n_players and len(game.target_sets) == n_players
    assert game.normalization_value == 0.0  # default value
    assert len(game.data_sets[0]) == pytest.approx(len(x_train) * player_sizes[0], abs=1)
    assert len(game.data_sets[1]) == pytest.approx(len(x_train) * player_sizes[1], abs=1)
    assert len(game.data_sets[2]) == pytest.approx(len(x_train) * player_sizes[2], abs=1)
    assert len(game.data_sets[3]) == pytest.approx(len(x_train) * player_sizes[3], abs=1)
    assert game_values[0] == 0.0
    assert len(game_values) == 4

    # test with classification dataset and changed input params
    x_train, y_train = background_clf_dataset
    x_test, y_test = background_clf_dataset
    model = DecisionTreeClassifier()

    # setup game
    game = DatasetValuation(
        x_train=x_train,
        y_train=y_train,
        fit_function=model.fit,
        predict_function=model.predict,
        loss_function=accuracy_score,
        n_players=n_players,
        random_state=42,
    )
    game_values = game(test_coalitions)
    assert game.n_players == n_players
    assert np.allclose(game.player_sizes, np.array([0.25, 0.25, 0.25, 0.25]))  # default is uniform
    assert len(game.data_sets) == n_players and len(game.target_sets) == n_players
    assert game_values[0] == 0.0
    assert len(game_values) == 4

    # check init with list of arrays
    x_train = [x_train] * n_players
    y_train = [y_train] * n_players

    game = DatasetValuation(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        fit_function=model.fit,
        predict_function=model.predict,
        loss_function=accuracy_score,
        n_players=n_players,
        random_state=42,
        empty_data_value=0.1,
        normalize=False,
    )
    game_values = game(test_coalitions)
    assert game.n_players == n_players
    assert game_values[0] == 0.1
    assert len(game_values) == 4

    # test storage and load
    game.precompute(test_coalitions)
    game.save_values("test_game.npz")
    assert os.path.exists("test_game.npz")
    _ = Game(path_to_values="test_game.npz")
    os.remove("test_game.npz")
    assert not os.path.exists("test_game.npz")

    # check for ValueError for missing params
    with pytest.raises(ValueError):
        _ = DatasetValuation(x_train=x_train, y_train=y_train)

    # check for ValueError with no x_test, y_test and x_train as list
    with pytest.raises(ValueError):
        _ = DatasetValuation(
            x_train=x_train,
            y_train=y_train,
            fit_function=model.fit,
            predict_function=model.predict,
            loss_function=accuracy_score,
            n_players=n_players,
            random_state=42,
        )

    # check the different player_sizes options

    # test with classification dataset and changed input params
    x_train, y_train = background_clf_dataset
    x_test, y_test = background_clf_dataset
    model = DecisionTreeClassifier()

    # increasing
    game = DatasetValuation(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        fit_function=model.fit,
        predict_function=model.predict,
        loss_function=accuracy_score,
        n_players=n_players,
        player_sizes="increasing",
        random_state=42,
    )
    assert list(game.player_sizes) == [0.1, 0.2, 0.3, 0.4]

    # uniform
    game = DatasetValuation(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        fit_function=model.fit,
        predict_function=model.predict,
        loss_function=accuracy_score,
        n_players=n_players,
        player_sizes="uniform",
        random_state=42,
    )
    assert list(game.player_sizes) == [0.25, 0.25, 0.25, 0.25]

    # random
    game = DatasetValuation(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        fit_function=model.fit,
        predict_function=model.predict,
        loss_function=accuracy_score,
        n_players=n_players,
        player_sizes="random",
        random_state=42,
    )
    assert len(game.player_sizes) == n_players
    assert np.sum(game.player_sizes) == 1.0

    # value error for wrong player_sizes string
    with pytest.raises(ValueError):
        _ = DatasetValuation(
            x_train=x_train,
            y_train=y_train,
            fit_function=model.fit,
            predict_function=model.predict,
            loss_function=accuracy_score,
            n_players=n_players,
            random_state=42,
            player_sizes="wrong",
        )


def test_california():
    """Tests the california housing Dataset Valuation Benchmark game."""

    game = CaliforniaHousingDatasetValuation()
    assert game.n_players == 10  # Default Value
    assert game.normalization_value == 0.0  # Default Value
    assert game.game_name == "CaliforniaHousing_DatasetValuation_Game"

    test_coalitions = np.zeros((2, 10), dtype=bool)
    test_coalitions[1, 0] = True
    test_coalitions[1, 1] = True

    game.precompute(test_coalitions)
    game.save_values("test_california_game.npz")
    assert os.path.exists("test_california_game.npz")
    _ = Game(path_to_values="test_california_game.npz")
    os.remove("test_california_game.npz")
    assert not os.path.exists("test_california_game.npz")

    # check for model loads
    game = CaliforniaHousingDatasetValuation(model_name="random_forest")
    assert game.n_players == 10

    with pytest.raises(ValueError):
        _ = CaliforniaHousingDatasetValuation(model_name="wrong_model")


def test_bike():
    """Tests the bike sharing Dataset Valuation Benchmark game."""

    game = BikeSharingDatasetValuation()
    assert game.n_players == 10  # Default Value
    assert game.normalization_value == 0.0  # Default Value
    assert game.game_name == "BikeSharing_DatasetValuation_Game"

    test_coalitions = np.zeros((2, 10), dtype=bool)
    test_coalitions[1, 0] = True
    test_coalitions[1, 1] = True

    game.precompute(test_coalitions)
    game.save_values("test_bike_game.npz")
    assert os.path.exists("test_bike_game.npz")
    _ = Game(path_to_values="test_bike_game.npz")
    os.remove("test_bike_game.npz")
    assert not os.path.exists("test_bike_game.npz")

    # check for model loads
    game = BikeSharingDatasetValuation(model_name="random_forest")
    assert game.n_players == 10

    with pytest.raises(ValueError):
        _ = BikeSharingDatasetValuation(model_name="wrong_model")


def test_adult_census():
    """Tests the adult census Dataset Valuation Benchmark game."""

    game = AdultCensusDatasetValuation()
    assert game.n_players == 10  # Default Value
    assert game.normalization_value == 0.0  # Default Value
    assert game.game_name == "AdultCensus_DatasetValuation_Game"

    test_coalitions = np.zeros((2, 10), dtype=bool)
    test_coalitions[1, 0] = True
    test_coalitions[1, 1] = True

    game.precompute(test_coalitions)
    game.save_values("test_adult_census_game.npz")
    assert os.path.exists("test_adult_census_game.npz")
    _ = Game(path_to_values="test_adult_census_game.npz")
    os.remove("test_adult_census_game.npz")

    # check for model loads
    game = AdultCensusDatasetValuation(model_name="random_forest")
    assert game.n_players == 10

    with pytest.raises(ValueError):
        _ = AdultCensusDatasetValuation(model_name="wrong_model")
