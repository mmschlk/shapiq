"""Pre-compute the values for the games and store them in a file."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from shapiq.games.base import Game

__all__ = [
    "SHAPIQ_DATA_DIR",
    "get_game_files",
    "pre_compute_and_store",
    "pre_compute_and_store_from_list",
    "pre_compute_from_configuration",
]

SHAPIQ_DATA_DIR: Path = Path(__file__).parent / "precomputed"
SHAPIQ_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_game_files(game: Game | Game.__class__ | str, n_players: int) -> list[str]:
    """Get the files for the given game and number of players.

    Args:
        game: The game to get the files for or the game name (as provided by `game.game_name` or
            `GameClass.get_game_name()`).
        n_players: The number of players in the game. If not provided, all files are returned.

    Returns:
        The list of files for the given game and number of players.

    """
    game_name = game
    if not isinstance(game, str):
        game_name = game.get_game_name()
    save_dir = Path(SHAPIQ_DATA_DIR) / game_name / str(n_players)
    try:
        return os.listdir(save_dir)
    except FileNotFoundError:
        return []


def pre_compute_and_store(
    game: Game,
    save_dir: str | None = None,
    game_id: str | None = None,
) -> Path:
    """Pre-compute the values for the given game and store them in a file.

    Args:
        game: The game to pre-compute the values for.
        save_dir: The path to the directory where the values are stored. If not provided, the
            directory is determined at random.
        game_id: A identifier of the game. If not provided, the ID is determined at random.

    Returns:
        The path to the file where the values are stored.

    """
    if save_dir is None:
        # this file path
        save_dir = Path(__file__).parent
        save_dir = save_dir / "precomputed" / game.game_name / str(game.n_players)
    elif str(game.n_players) not in save_dir:
        save_dir = Path(save_dir) / str(game.n_players)
    save_dir.mkdir(parents=True, exist_ok=True)

    if game_id is None:
        game_id = str(hash(game))[:8]

    save_path = Path(save_dir) / game_id

    game.precompute()
    game.save_values(path=save_path)
    return save_path


def pre_compute_from_configuration(
    game_class: Game.__class__ | str,
    configuration: dict[str, Any] | None = None,
    n_iterations: int | None = None,
    n_player_id: int = 0,
    n_jobs: int = 1,
) -> list[str]:
    """Pre-compute the game data for the given game class and configuration.

    Pre-compute the game data for the given game class and configuration if it is not already
    pre-computed.

    This function will pre-compute the game data for the given game class and configuration. The
    game data will be stored in the `SHAPIQ_DATA_DIR` directory in a subdirectory with the game name
    and player size. The file name will be generated based on the configuration and iteration.
    """
    from .configuration import (
        BENCHMARK_CONFIGURATIONS,
        BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS,
        BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS,
        get_game_class_from_name,
        get_game_file_name_from_config,
    )

    game_class = get_game_class_from_name(game_class) if isinstance(game_class, str) else game_class

    show_tqdm = True

    game_class_config = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]
    n_players = game_class_config["n_players"]
    iteration_parameter = game_class_config["iteration_parameter"]
    configurations = [configuration]
    default_params = BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS
    if default_params["verbose"] is True:
        show_tqdm = False
    if configuration is None:
        configurations = game_class_config["configurations"]

    iterations = game_class_config.get(
        "iteration_parameter_values",
        BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS,
    )
    iteration_names = game_class_config.get("iteration_parameter_values_names", iterations)
    if n_iterations is not None:
        iterations = iterations[:n_iterations]
        iteration_names = iteration_names[:n_iterations]

    parameter_space = []
    for config in configurations:
        for iteration, iteration_name in zip(iterations, iteration_names, strict=False):
            save_dir = Path(SHAPIQ_DATA_DIR) / game_class.get_game_name() / str(n_players)
            game_id = get_game_file_name_from_config(config, iteration)
            save_path = Path(save_dir) / game_id
            save_path_npz = save_path.with_suffix(".npz")
            save_path_csv = save_path.with_suffix(".csv")
            if save_path.exists() or save_path_npz.exists() or save_path_csv.exists():
                continue

            params = default_params.copy()
            params.update(config)
            params[iteration_parameter] = iteration_name
            parameter_space.append((params, save_dir, game_id))

    created_files = []
    if n_jobs == 1:
        parameter_generator = tqdm(parameter_space) if show_tqdm else parameter_space
        for params, save_dir, game_id in parameter_generator:
            game = game_class(**params)
            save_path = pre_compute_and_store(game, save_dir, game_id)
            created_files.append(save_path)
    else:
        with mp.Pool(n_jobs) as pool:
            results = list(
                pool.starmap(
                    pre_compute_and_store,
                    [
                        (game_class(**params), save_dir, game_id)
                        for params, save_dir, game_id in parameter_space
                    ],
                ),
            )
            created_files.extend(results)

    return created_files


def pre_compute_and_store_from_list(
    games: list[Game],
    save_dir: str | None = None,
    game_ids: list[str] | None = None,
    n_jobs: int = 1,
) -> list[str]:
    """Pre-compute the values for the games stored in the given file.

    Args:
        games: The games to pre-compute the values for.
        save_dir: The path to the directory where the values are stored. If not provided, the
            directory is determined at random.
        game_ids: The IDs of the games. If not provided, the IDs are determined at random.
        n_jobs: The number of parallel jobs to run.

    Returns:
        The paths to the files where the values are stored.

    """
    if game_ids is None:
        game_ids = [None] * len(games)

    if n_jobs == 1:
        return [
            pre_compute_and_store(game, save_dir, game_id)
            for game, game_id in zip(games, game_ids, strict=False)
        ]

    with mp.Pool(n_jobs) as pool:
        return list(
            tqdm(
                pool.starmap(
                    pre_compute_and_store,
                    [
                        (game, save_dir, game_id)
                        for game, game_id in zip(games, game_ids, strict=False)
                    ],
                ),
                total=len(games),
            ),
        )
