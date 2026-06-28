"""Game factory for the leaderboard runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import shapiq_games.benchmark.global_xai.benchmark_tabular as global_tabular
import shapiq_games.benchmark.local_xai.benchmark_tabular as local_tabular
from leaderboard.config_manager.constants import (
    GLOBAL_GAME_REGISTRY,
    LOCAL_GAME_REGISTRY,
)
from leaderboard.runner.runner_exceptions import UnknownGameError
from shapiq_games.synthetic import SOUM

if TYPE_CHECKING:
    from shapiq import Game


def create_game_from_config(
    run_config: dict[str, Any],
    base_config: dict[str, Any],
) -> tuple[Game, dict[str, Any]]:
    """Create a game instance from benchmark configuration dictionaries, routing between local and global XAI definitions.

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
    game_family = base_config.get("game_family", "local_xai")

    # 1. Build a streamlined fallback dictionary based ONLY on the family pipeline type
    if game_name == "SOUM":
        default_game_params = {
            "n": 10,
            "n_basis_games": 20,
            "min_interaction_size": 1,
            "max_interaction_size": max_order,
            "random_state": game_seed,
        }
    else:
        # Shared defaults by ALL tabular machine learning games
        default_game_params = {
            "model_name": "decision_tree",
            "normalize": True,
            "verbose": False,
            "random_state": game_seed,
        }

        # Inject local runner pipeline defaults only if family is local_xai
        if game_family == "local_xai":
            default_game_params["x"] = run_config.get("x", 0)
            default_game_params["imputer"] = "marginal"
            # default_game_params["class_to_explain"] = run_config.get("class_to_explain")

    # 2. Merge dictionaries. base_config (fully sanitized by Pydantic) has supreme priority!
    game_params = {
        **default_game_params,
        **base_config.get("game_params", {}),
    }

    # Centralized path resolution for visual games
    if game_name == "ImageClassifier" and "x_explain_path" in game_params:
        project_root = Path(__file__).resolve().parents[3]
        raw_img_path = Path(game_params["x_explain_path"])

        # If path is relative,then append project_root
        if not raw_img_path.is_absolute():
            absolute_img_path = project_root / raw_img_path
            game_params["x_explain_path"] = str(absolute_img_path)

            if not absolute_img_path.exists():
                raise FileNotFoundError(f"Image not found at: {absolute_img_path}")

    if game_name == "SOUM":
        game_class = SOUM
    elif game_family == "global_xai":
        if game_name not in GLOBAL_GAME_REGISTRY:
            raise UnknownGameError(
                f"Game '{game_name}' is not supported in global_xai family. "
                f"Available global games are: {tuple(GLOBAL_GAME_REGISTRY.keys())}"
            )
        game_class = GLOBAL_GAME_REGISTRY[game_name]
    else:
        game_class = LOCAL_GAME_REGISTRY[game_name]

    game = game_class(**game_params)

    return game, game_params
