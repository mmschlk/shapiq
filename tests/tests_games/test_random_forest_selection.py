"""This test module tests the ensemble selection games."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    f1_score,
)

from shapiq.games.benchmark import RandomForestEnsembleSelection, EnsembleSelection
from shapiq.games.benchmark import (
    AdultCensusRandomForestEnsembleSelection,
    BikeSharingRandomForestEnsembleSelection,
    CaliforniaHousingRandomForestEnsembleSelection,
)
from shapiq.games import Game


@pytest.mark.parametrize(
    "task, loss_function",
    [
        ("regression", "r2_score"),
        ("regression", "mean_squared_error"),
        ("regression", "mean_absolute_error"),
        ("classification", "accuracy_score"),
        ("classification", "f1_score"),
    ],
)
def test_random_forest_selection(
    task,
    loss_function,
    background_reg_dataset,
    background_clf_dataset_binary,
    rf_clf_binary_model,
    rf_reg_model,
):
    """Tests the EnsembleSelection game on a regression and classification dataset."""
    # start with regression data
    if task == "regression":
        data, target = background_reg_dataset
        model = rf_reg_model
    else:
        data, target = background_clf_dataset_binary
        model = rf_clf_binary_model
    n_players = len(model.estimators_)

    if loss_function == "r2_score":
        loss_function = r2_score
    elif loss_function == "accuracy_score":
        loss_function = accuracy_score
    elif loss_function == "mean_squared_error":
        loss_function = mean_squared_error
    elif loss_function == "mean_absolute_error":
        loss_function = mean_absolute_error
    elif loss_function == "f1_score":
        loss_function = f1_score

    game = RandomForestEnsembleSelection(
        random_forest=model,
        x_train=data,
        y_train=target,
        x_test=data,
        y_test=target,
        loss_function=loss_function,
        dataset_type=task,
        verbose=False,
        normalize=True,
    )
    assert isinstance(game, Game)
    assert isinstance(game, EnsembleSelection)
    assert isinstance(game, RandomForestEnsembleSelection)
    assert game.n_players == n_players

    test_coalitions = np.array([[False] * n_players, [True] * n_players])
    values = game(test_coalitions)
    assert values.shape == (2,)
    assert values[0] == 0.0
    assert values[1] != 0.0


def test_adult():
    """Test the AdultCensus random forest selection game."""
    n_players = 10
    test_coalitions = np.array([[False] * n_players, [True] * n_players])
    game = AdultCensusRandomForestEnsembleSelection()
    values = game(test_coalitions)
    assert isinstance(game, Game)
    assert isinstance(game, EnsembleSelection)
    assert game.n_players == n_players
    assert values.shape == (2,)
    assert values[0] == 0.0  # should be zero
    assert values[1] != 0.0  # should not be zero
    assert game.game_name == "AdultCensus_RandomForestEnsembleSelection_EnsembleSelection_Game"


def test_california():
    """Test the CaliforniaHousing ensemble selection game."""
    n_players = 10
    test_coalitions = np.array([[False] * n_players, [True] * n_players])
    game = CaliforniaHousingRandomForestEnsembleSelection()
    values = game(test_coalitions)
    assert isinstance(game, Game)
    assert isinstance(game, EnsembleSelection)
    assert game.n_players == n_players
    assert values.shape == (2,)
    assert values[0] == 0.0  # should be zero
    assert values[1] != 0.0  # should not be zero
    assert (
        game.game_name == "CaliforniaHousing_RandomForestEnsembleSelection_EnsembleSelection_Game"
    )


def test_bike():
    """Test the BikeSharing ensemble selection game."""
    n_players = 10
    test_coalitions = np.array([[False] * n_players, [True] * n_players])
    game = BikeSharingRandomForestEnsembleSelection()
    values = game(test_coalitions)
    assert isinstance(game, Game)
    assert isinstance(game, EnsembleSelection)
    assert game.n_players == n_players
    assert values.shape == (2,)
    assert values[0] == 0.0  # should be zero
    assert values[1] != 0.0  # should not be zero
    assert game.game_name == "BikeSharing_RandomForestEnsembleSelection_EnsembleSelection_Game"
