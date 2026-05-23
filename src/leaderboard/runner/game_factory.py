"""Game factory for the leaderboard runner."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from leaderboard.runner.runner_exceptions import UnknownGameError
from shapiq_games.benchmark.local_xai.benchmark_tabular import (
    AdultCensus,
    Annealing,
    Arrhythmia,
    BikeSharing,
    BreastCancer,
    CaliforniaHousing,
    Hepatitis,
    Ionosphere,
    Mushroom,
    Nursery,
    Soybean,
    Thyroid,
    Zoo,
)
from shapiq_games.synthetic import SOUM

if TYPE_CHECKING:
    from shapiq import Game


GAME_REGISTRY = {
    "SOUM": SOUM,
    "BikeSharing": BikeSharing,
    "CaliforniaHousing": CaliforniaHousing,
    "AdultCensus": AdultCensus,
    "Mushroom": Mushroom,
    "Soybean": Soybean,
    "Thyroid": Thyroid,
    "Annealing": Annealing,
    "Arrhythmia": Arrhythmia,
    "BreastCancer": BreastCancer,
    "Hepatitis": Hepatitis,
    "Ionosphere": Ionosphere,
    "Nursery": Nursery,
    "Zoo": Zoo,
}

REGRESSION_GAMES = {
    "BikeSharing",
    "CaliforniaHousing",
}

CLASSIFICATION_GAMES = {
    "AdultCensus",
    "Mushroom",
    "Soybean",
    "Thyroid",
    "Annealing",
    "Arrhythmia",
    "BreastCancer",
    "Hepatitis",
    "Ionosphere",
    "Nursery",
    "Zoo",
}


def create_game_from_config(
    run_config: dict[str, Any],
    base_config: dict[str, Any],
) -> tuple[Game, dict[str, Any]]:
    """Create a game instance from benchmark configuration dictionaries.

    The function reads the selected game from ""run_config"" and uses
    ""base_config"" to override default game parameters if "game_params" is
    provided.

    Args:
        run_config: The concrete run configuration.
        base_config: The original benchmark configuration.

    Returns:
        A tuple containing the created game instance and the game parameters
        used to initialize it.

    Raises:
        KeyError: If required entries are missing.
        UnknownGameError: If the configured game is unknown.
    """
    game_name = run_config["game"]
    game_seed = run_config["game_seed"]
    max_order = run_config["max_order"]

    if game_name == "SOUM":
        default_game_params = {
            "n": 10,
            "n_basis_games": 20,
            "min_interaction_size": 1,
            "max_interaction_size": max_order,
            "random_state": game_seed,
        }

    elif game_name in REGRESSION_GAMES:
        default_game_params = {
            "x": run_config.get("x", 0),
            "model_name": "decision_tree",
            "imputer": "marginal",
            "normalize": True,
            "verbose": False,
            "random_state": game_seed,
        }

    elif game_name in CLASSIFICATION_GAMES:
        default_game_params = {
            "x": run_config.get("x", 0),
            "class_to_explain": run_config.get("class_to_explain", None),
            "model_name": "decision_tree",
            "imputer": "marginal",
            "normalize": True,
            "verbose": False,
            "random_state": game_seed,
        }

    else:
        available = ", ".join(GAME_REGISTRY)
        raise UnknownGameError(f"Unknown game: {game_name}. Available games: {available}")

    game_params = {
        **default_game_params,
        **base_config.get("game_params", {}),
    }

    game_class = GAME_REGISTRY[game_name]
    game = game_class(**game_params)

    return game, game_params