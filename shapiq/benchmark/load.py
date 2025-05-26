"""Utility functions to load benchmark games.

This module contains all utility functions to load benchmark games from the configurations or
from the precomputed data (GitHub repository).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from shapiq.benchmark.configuration import (
    BENCHMARK_CONFIGURATIONS,
    BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS,
    BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS,
    GAME_NAME_TO_CLASS_MAPPING,
    get_game_file_name_from_config,
)
from shapiq.benchmark.precompute import SHAPIQ_DATA_DIR
from shapiq.games.base import Game

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "download_game_data",
    "load_game_data",
    "load_games_from_configuration",
]


def load_games_from_configuration(
    game_class: Game.__class__ | str,
    config_id: int | dict[str, Any],
    *,
    n_games: int | None = None,
    n_player_id: int = 0,
    check_pre_computed: bool = True,
    only_pre_computed: bool = True,
) -> Generator[Game, None, None]:
    """Load the game with the given configuration from disk or create it if it does not exist.

    Args:
        game_class: The class of the game to load with the configuration.
        config_id: The configuration to use to load the game.
        n_games: The number of games to load. Defaults to None.
        n_player_id: The player ID to use. Defaults to 0. Not all games have multiple player IDs.
        check_pre_computed: A flag to check if the game is pre-computed (load from disk). Defaults
            to True.
        only_pre_computed: A flag to only load the pre-computed games. Defaults to False.

    Returns:
        An initialized game object with the given configuration.

    """
    game_class = (
        GAME_NAME_TO_CLASS_MAPPING[game_class] if isinstance(game_class, str) else game_class
    )
    # get config if it is an int
    try:
        configuration: dict = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]["configurations"][
            config_id - 1
        ]
    except TypeError:  # not a dict
        configuration: dict = config_id
    params = {}

    # get the default parameters
    default_params = BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS.copy()
    params.update(default_params)
    params.update(configuration)

    # get the class-specific configurations of how the iterations are set up
    config_of_class = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]
    game_should_be_precomputed = config_of_class["precompute"]
    iteration_param = config_of_class["iteration_parameter"]
    iteration_param_values = config_of_class.get(
        "iteration_parameter_values",
        BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS,
    )
    iteration_param_values_names = config_of_class.get(
        "iteration_parameter_values_names",
        iteration_param_values,
    )

    # create the generator of games
    n_games = (
        len(iteration_param_values)
        if n_games is None
        else min(n_games, len(iteration_param_values))
    )
    for i in range(n_games):
        game_iteration = iteration_param_values[i]  # from 1 to 30
        game_iteration_value = iteration_param_values_names[i]  # i.e. the sentence or random state
        params[iteration_param] = game_iteration_value  # set the iteration parameter
        if not game_should_be_precomputed or (
            not check_pre_computed and not only_pre_computed
        ):  # e.g. for SynthDataTreeSHAPIQXAI
            yield game_class(**params)
        else:
            try:  # try to load the game from disk
                yield load_game_data(
                    game_class,
                    configuration,
                    iteration=game_iteration,
                    n_player_id=n_player_id,
                )
            except FileNotFoundError:
                if only_pre_computed:  # if only pre-computed games are requested, skip the game
                    continue
                else:  # fallback to creating the game which is not pre-computed
                    yield game_class(**params)


def load_game_data(
    game_class: Game.__class__,
    configuration: dict[str, Any],
    iteration: int = 1,
    n_player_id: int = 0,
) -> Game:
    """Loads the precomputed game data for the given game class and configuration.

    Args:
        game_class: The class of the game
        configuration: The configuration to use to load the game
        iteration: The iteration of the game to load
        n_player_id: The player ID to use. Defaults to 0. Not all games have multiple player IDs.

    Returns:
        An initialized game object with the given configuration

    Raises:
        FileNotFoundError: If the file with the precomputed values does not exist

    """
    n_players = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]["n_players"]
    file_name = get_game_file_name_from_config(configuration, iteration)

    path_to_values = (
        SHAPIQ_DATA_DIR / game_class.get_game_name() / str(n_players) / f"{file_name}.npz"
    )
    try:
        return Game(
            path_to_values=path_to_values,
            verbose=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"],
            normalize=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["normalize"],
        )
    except FileNotFoundError:
        # download the game data if it does not exist
        download_game_data(game_class.get_game_name(), n_players, file_name)
        try:
            return Game(
                path_to_values=path_to_values,
                verbose=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"],
                normalize=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["normalize"],
            )
        except FileNotFoundError as error:
            msg = (
                f"Game data for game {game_class.get_game_name()} with configuration "
                f"{configuration} and iteration {iteration} could not be found."
            )
            raise FileNotFoundError(msg) from error


def download_game_data(game_name: str, n_players: int, file_name: str) -> None:
    """Downloads the game file from the repository.

    Args:
        game_name: The name of the game.
        n_players: The number of players in the game.
        file_name: The name of the file to download.

    Raises:
        FileNotFoundError: If the file could not be downloaded.

    """
    github_url = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/precomputed_games"

    # create the directory if it does not exist
    game_dir = SHAPIQ_DATA_DIR / game_name / str(n_players)
    game_dir.mkdir(parents=True, exist_ok=True)

    # download the file
    file_name = file_name.replace(".npz", "")
    path = Path(game_dir) / f"{file_name}.npz"
    url = f"{github_url}/{game_name}/{n_players}/{file_name}.npz"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        msg = f"Could not download the game data from {url}. Check if configuration is correct."
        raise FileNotFoundError(msg) from error
    with Path(path).open("wb") as file:
        file.write(response.content)
        time.sleep(0.01)
