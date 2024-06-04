"""Pre-compute the values for the games and store them in a file."""

import multiprocessing as mp
import os
from typing import Any, Optional, Union

from tqdm.auto import tqdm

from ..base import Game

__all__ = [
    "pre_compute_from_configuration",
    "pre_compute_and_store",
    "pre_compute_and_store_from_list",
    "SHAPIQ_DATA_DIR",
    "get_game_files",
]


SHAPIQ_DATA_DIR = os.path.join(os.path.dirname(__file__), "precomputed")
os.makedirs(SHAPIQ_DATA_DIR, exist_ok=True)

GITHUB_GAMES_DATA_URL = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/precomputed/"


def download_precomputed_data(file_name: str):
    """Download the pre-computed data benchmarks games."""
    pass


def get_game_files(game: Union[Game, Game.__class__, str], n_players: int) -> list[str]:
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
    save_dir = os.path.join(SHAPIQ_DATA_DIR, game_name, str(n_players))
    try:
        return os.listdir(save_dir)
    except FileNotFoundError:
        return []


def pre_compute_and_store(
    game: Game, save_dir: Optional[str] = None, game_id: Optional[str] = None
) -> str:
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
        save_dir = os.path.dirname(__file__)
        save_dir = os.path.join(save_dir, "precomputed", game.game_name, str(game.n_players))
    else:  # check if n_players is in the path
        if str(game.n_players) not in save_dir:
            save_dir = os.path.join(save_dir, str(game.n_players))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if game_id is None:
        game_id = str(hash(game))[:8]

    save_path = os.path.join(save_dir, game_id)

    game.precompute()
    game.save_values(path=save_path)
    return save_path


def pre_compute_from_configuration(
    game_class: Union[Game.__class__, str],
    configuration: Optional[dict[str, Any]] = None,
    n_iterations: Optional[int] = None,
    n_player_id: int = 0,
    n_jobs: int = 1,
) -> list[str]:
    """Pre-compute the game data for the given game class and configuration if it is not already
    pre-computed.

    This function will pre-compute the game data for the given game class and configuration. The
    game data will be stored in the `SHAPIQ_DATA_DIR` directory in a subdirectory with the game name
    and player size. The file name will be generated based on the configuration and iteration.
    """
    from .benchmark_config import (
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
        "iteration_parameter_values", BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS
    )
    iteration_names = game_class_config.get("iteration_parameter_values_names", iterations)
    if n_iterations is not None:
        iterations = iterations[:n_iterations]
        iteration_names = iteration_names[:n_iterations]

    parameter_space = []
    for config in configurations:
        print(
            f"Pre-computing game data for {game_class.get_game_name()}, "
            f"configuration: {config}, n_players: {n_players}, iterations: {iterations}, "
            f"iteration names: {iteration_names}"
        )

        for iteration, iteration_name in zip(iterations, iteration_names):
            save_dir = os.path.join(SHAPIQ_DATA_DIR, game_class.get_game_name(), str(n_players))
            game_id = get_game_file_name_from_config(config, iteration)
            save_path = os.path.join(save_dir, game_id)

            if (
                os.path.exists(save_path)
                or os.path.exists(save_path + ".npz")
                or os.path.exists(save_path + ".csv")
            ):
                print(f"Game data for {game_class.get_game_name()} already pre-computed.")
                continue

            params = default_params.copy()
            params.update(config)
            params[iteration_parameter] = iteration_name
            parameter_space.append((params, save_dir, game_id))

    created_files = []
    if n_jobs == 1:
        print(f"Pre-computing game data for {len(parameter_space)} configurations in sequence.")
        parameter_generator = tqdm(parameter_space) if show_tqdm else parameter_space
        for params, save_dir, game_id in parameter_generator:
            game = game_class(**params)
            save_path = pre_compute_and_store(game, save_dir, game_id)
            created_files.append(save_path)
    else:
        print(f"Pre-computing game data for {len(parameter_space)} configurations in parallel.")
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
    save_dir: Optional[str] = None,
    game_ids: Optional[list[str]] = None,
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
            pre_compute_and_store(game, save_dir, game_id) for game, game_id in zip(games, game_ids)
        ]

    with mp.Pool(n_jobs) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    pre_compute_and_store,
                    [(game, save_dir, game_id) for game, game_id in zip(games, game_ids)],
                ),
                total=len(games),
            )
        )

    return results
