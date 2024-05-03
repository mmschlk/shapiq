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

from shapiq.games.benchmark import EnsembleSelection
from shapiq.games.benchmark import (
    AdultCensusEnsembleSelection,
    BikeSharingEnsembleSelection,
    CaliforniaHousingEnsembleSelection,
)
from shapiq.games import Game


# with this set tests take around 1s
ENSEMBLE_MEMBERS_FAST = ["regression", "decision_tree", "random_forest", "gradient_boosting", "knn"]
ENSEMBLE_MEMBERS_VERY_FAST = ["regression", "decision_tree", "regression", "decision_tree"]


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
def test_ensemble_selection(
    task, loss_function, background_reg_dataset, background_clf_dataset_binary
):
    """Tests the EnsembleSelection game on a regression and classification dataset."""
    # start with regression data
    if task == "regression":
        data, target = background_reg_dataset
    else:
        data, target = background_clf_dataset_binary

    n_players = len(ENSEMBLE_MEMBERS_VERY_FAST)
    ensemble_members = ENSEMBLE_MEMBERS_VERY_FAST

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

    game = EnsembleSelection(
        x_train=data,
        y_train=target,
        x_test=data,
        y_test=target,
        loss_function=loss_function,
        dataset_type=task,
        ensemble_members=ensemble_members,
        n_members=n_players,
        verbose=False,
        random_state=42,
        normalize=True,
    )
    assert isinstance(game, Game)
    assert isinstance(game, EnsembleSelection)
    assert game.n_players == n_players

    test_coalitions = np.array([[False] * n_players, [True] * n_players])
    values = game(test_coalitions)
    assert values.shape == (2,)
    assert values[0] == 0.0
    assert values[1] != 0.0


def test_adult():
    """Test the AdultCensus ensemble selection game."""
    ensemble_members = ENSEMBLE_MEMBERS_FAST
    n_players = len(ensemble_members)
    test_coalitions = np.array([[False] * n_players, [True] * n_players])
    game = AdultCensusEnsembleSelection(ensemble_members=ensemble_members)
    values = game(test_coalitions)
    assert isinstance(game, Game)
    assert isinstance(game, EnsembleSelection)
    assert game.n_players == n_players
    assert values.shape == (2,)
    assert values[0] == 0.0  # should be zero
    assert values[1] != 0.0  # should not be zero
    assert game.game_name == "AdultCensus_EnsembleSelection_Game"


def test_california():
    """Test the CaliforniaHousing ensemble selection game."""
    ensemble_members = ENSEMBLE_MEMBERS_FAST
    n_players = len(ensemble_members)
    test_coalitions = np.array([[False] * n_players, [True] * n_players])
    game = CaliforniaHousingEnsembleSelection(ensemble_members=ensemble_members)
    values = game(test_coalitions)
    assert isinstance(game, Game)
    assert isinstance(game, EnsembleSelection)
    assert game.n_players == n_players
    assert values.shape == (2,)
    assert values[0] == 0.0  # should be zero
    assert values[1] != 0.0  # should not be zero
    assert game.game_name == "CaliforniaHousing_EnsembleSelection_Game"


def test_bike():
    """Test the BikeSharing ensemble selection game."""
    ensemble_members = ENSEMBLE_MEMBERS_FAST
    n_players = len(ensemble_members)
    test_coalitions = np.array([[False] * n_players, [True] * n_players])
    game = BikeSharingEnsembleSelection(ensemble_members=ensemble_members)
    values = game(test_coalitions)
    assert isinstance(game, Game)
    assert isinstance(game, EnsembleSelection)
    assert game.n_players == n_players
    assert values.shape == (2,)
    assert values[0] == 0.0  # should be zero
    assert values[1] != 0.0  # should not be zero
    assert game.game_name == "BikeSharing_EnsembleSelection_Game"
