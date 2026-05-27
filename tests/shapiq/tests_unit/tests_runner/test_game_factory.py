from __future__ import annotations

import pytest

from leaderboard.runner.game_factory import create_game_from_config
from leaderboard.runner.runner_exceptions import UnknownGameError
from shapiq_games.benchmark.local_xai import CaliforniaHousing
from shapiq_games.synthetic import SOUM


def test_create_game_from_config():
    """Test that the game factory creates a SOUM game with default parameters."""
    run_config = {
        "game": "SOUM",
        "game_seed": 42,
        "max_order": 2,
    }

    base_config = {}

    game, game_params = create_game_from_config(
        run_config=run_config,
        base_config=base_config,
    )

    assert isinstance(game, SOUM)

    assert game_params == {
        "n": 10,
        "n_basis_games": 20,
        "min_interaction_size": 1,
        "max_interaction_size": 2,
        "random_state": 42,
    }

    assert game.n_players == 10


def test_create_game_from_config_with_params_override():
    """Test that game_params from base_config override defaults."""
    run_config = {
        "game": "SOUM",
        "game_seed": 42,
        "max_order": 2,
    }

    base_config = {
        "game_params": {
            "n": 5,
            "n_basis_games": 3,
            "min_interaction_size": 1,
            "max_interaction_size": 2,
            "random_state": 123,
        },
    }

    game, game_params = create_game_from_config(
        run_config=run_config,
        base_config=base_config,
    )

    assert isinstance(game, SOUM)
    assert game.n_players == 5
    assert game_params["n"] == 5
    assert game_params["n_basis_games"] == 3
    assert game_params["random_state"] == 123


def test_create_game_from_config_creates_california_housing():
    """Test that the game factory creates a CaliforniaHousing game with default parameters."""
    run_config = {
        "game": "CaliforniaHousing",
        "game_seed": 42,
        "max_order": 1,
    }

    base_config = {}

    game, game_params = create_game_from_config(
        run_config=run_config,
        base_config=base_config,
    )

    assert isinstance(game, CaliforniaHousing)
    assert game.n_players == 8

    assert game_params["x"] == 0
    assert game_params["model_name"] == "decision_tree"
    assert game_params["imputer"] == "marginal"
    assert game_params["normalize"] is True
    assert game_params["verbose"] is False
    assert game_params["random_state"] == 42


def test_create_game_from_config_raises_for_unknown_game():
    """Test that an unknown game name raises UnknownGameError."""
    run_config = {
        "game": "UnknownGame",
        "game_seed": 42,
        "max_order": 1,
    }

    base_config = {}

    with pytest.raises(UnknownGameError):
        create_game_from_config(
            run_config=run_config,
            base_config=base_config,
        )
