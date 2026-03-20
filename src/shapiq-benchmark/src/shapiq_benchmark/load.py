"""Utility functions to load benchmark games.

This module contains all utility functions to load benchmark games from the configurations or
from the precomputed data (GitHub repository).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from shapiq.game import Game

from .configuration import (
    BENCHMARK_CONFIGURATIONS,
    BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS,
    BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS,
    GAME_NAME_TO_CLASS_MAPPING,
    get_game_file_name_from_config,
)
from .precompute import SHAPIQ_DATA_DIR

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "download_game_data",
    "load_game_data",
    "load_games_from_configuration",
]


class GameFactory:
    """A factory to create game instances from configuration files."""

    @staticmethod
    def create_game_from_configs(
        configuration: dict[str, Any],
        game_should_be_precomputed: bool,
        iteration_param: str,
        iteration_param_values: list[Any],
        iteration_param_values_names: list[Any],
        game_class: Game,
        n_players: int,
        *,
        n_games: int | None = None,
        check_pre_computed: bool = True,
        only_pre_computed: bool = True,
    ) -> Generator[Game, None, None]:
        """Creates a game instance from the given configuration.

        Args:
            configuration: The configuration to use to create the game.
            precompute: A flag to indicate if the game should be pre-computed.
            iteration_parameter: The parameter to iterate over.
            iteration_parameter_values: The values of the iteration parameter.
            n_players: The number of players in the game.
            n_games: The number of games to create. Defaults to None.
            check_pre_computed: A flag to check if the game is pre-computed (load from disk). Defaults
                to True.
            only_pre_computed: A flag to only load the pre-computed games. Defaults to False.

        Returns:
            An initialized game object with the given configuration.
        """
        params = BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS.copy()
        params.update(configuration)
        ## Create the generator of games
        n_games = (
            len(iteration_param_values)
            if n_games is None
            else min(n_games, len(iteration_param_values))
        )
        for i in range(n_games):
            game_iteration = iteration_param_values[i]  # from 1 to 30
            game_iteration_value = iteration_param_values_names[
                i
            ]  # i.e. the sentence or random state
            params[iteration_param] = game_iteration_value  # set the iteration parameter
            if not game_should_be_precomputed or (
                not check_pre_computed and not only_pre_computed
            ):  # e.g. for SynthDataTreeSHAPIQXAI
                # print(f"Creating game {game_class.get_game_name()} with params: {params}")
                yield game_class(**params)
            else:
                try:  # try to load the game from disk
                    yield GameFactory.load_game_from_precomputed(
                        game_class,
                        n_players,
                        configuration,
                        iteration=game_iteration,
                    )
                except FileNotFoundError:
                    # Try loading .json files
                    try:  # try to load the game from disk
                        yield GameFactory.load_game_from_precomputed_json(
                            game_class,
                            n_players,
                            configuration,
                            iteration=game_iteration,
                        )
                    except FileNotFoundError:
                        if (
                            only_pre_computed
                        ):  # if only pre-computed games are requested, skip the game
                            continue
                        else:  # fallback to creating the game which is not pre-computed
                            yield game_class(**params)

    @staticmethod
    def load_game_from_precomputed(
        game_class: Game,
        n_players: int,
        configuration: dict[str, Any],
        iteration: int = 1,
    ) -> Game:
        """Loads a precomputed game from disk.

        Args:
            game_class: The class of the game to load.
            configuration: The configuration to use to load the game.
            iteration: The iteration of the game to load. Defaults to 1.
            n_player_id: The player ID to use. Defaults to 0. Not all games have multiple player IDs.

        Returns:
            An initialized game object with the given configuration.
        """
        file_name = get_game_file_name_from_config(configuration, iteration)
        path_to_values = (
            SHAPIQ_DATA_DIR / game_class.get_game_name() / str(n_players) / f"{file_name}.npz"
        )

        try:
            game = Game(
                n_players=n_players,
                normalize=False,
                verbose=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"],
            )
            game.load_values(path_to_values)
            return game
        except FileNotFoundError:
            # download the game data if it does not exist
            download_game_data(game_class.get_game_name(), n_players, file_name)
            try:
                game = Game(
                    n_players=n_players,
                    normalize=False,
                    verbose=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"],
                )
                game.load_values(path_to_values)
                return game
            except FileNotFoundError as error:
                msg = (
                    f"Game data for game {game_class.get_game_name()} with configuration "
                    f"{configuration} and iteration {iteration} could not be found."
                )
                raise FileNotFoundError(msg) from error

    @staticmethod
    def load_game_from_precomputed_json(
        game_class: Game,
        n_players: int,
        configuration: dict[str, Any],
        iteration: int = 1,
    ) -> Game:
        """Loads a precomputed game from disk.

        Args:
            game_class: The class of the game to load.
            configuration: The configuration to use to load the game.
            iteration: The iteration of the game to load. Defaults to 1.
            n_player_id: The player ID to use. Defaults to 0. Not all games have multiple player IDs.

        Returns:
            An initialized game object with the given configuration.
        """
        file_name = get_game_file_name_from_config(configuration, iteration)
        # print(f"Loading precomputed game {game_class.get_game_name()} with file name: {file_name}")
        path_to_values = (
            SHAPIQ_DATA_DIR / game_class.get_game_name() / str(n_players) / f"{file_name}.json"
        )
        # print(path_to_values)

        try:
            return Game.from_json_file(path_to_values)
        except FileNotFoundError as error:
            msg = (
                f"Game data for game {game_class.get_game_name()} with configuration "
                f"{configuration} and iteration {iteration} could not be found."
            )
            raise FileNotFoundError(msg) from error

    @staticmethod
    def load_configuration_file_interactive(
        config_path: str,
        *,
        n_games: int | None = None,
        check_pre_computed: bool = True,
        only_pre_computed: bool = True,
        return_config_id: bool = False,
    ) -> Generator[Game, None, None]:
        """Loads a configuration file from the given path.

        Args:
            config_path: The path to the configuration file.

        Returns:
            The loaded configuration as a dictionary.
        """
        params = BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS.copy()
        with open(config_path) as file:
            configurations = json.load(file)
        game_class: Game = GAME_NAME_TO_CLASS_MAPPING[configurations["game_name"]]

        ## Load Configuration ID
        # Check if there is only one configuration
        if len(configurations["configurations"]) == 1:
            config_id = 1
        else:
            for _idx, _config in enumerate(configurations["configurations"], start=1):
                pass
            config_id = int(
                input("Enter configuration ID (int) or 'dict' to use the full configuration: ")
            )
        # Load the configuration
        if config_id > 0 and config_id <= len(configurations["configurations"]):
            configuration = configurations["configurations"][config_id - 1]
        else:
            msg = "Invalid configuration ID."
            raise ValueError(msg)
        params.update(configuration)
        ## Load class-specific configurations
        game_should_be_precomputed = configurations["precompute"]
        iteration_param = configurations["iteration_parameter"]
        iteration_param_values = configurations.get(
            "iteration_parameter_values",
            BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS,
        )
        iteration_param_values_names = configurations.get(
            "iteration_parameter_values_names", iteration_param_values
        )

        game_generator = GameFactory.create_game_from_configs(
            configuration=configuration,
            game_should_be_precomputed=game_should_be_precomputed,
            iteration_param=iteration_param,
            iteration_param_values=iteration_param_values,
            iteration_param_values_names=iteration_param_values_names,
            game_class=game_class,
            n_players=configurations["n_players"],
            n_games=n_games,
            check_pre_computed=check_pre_computed,
            only_pre_computed=only_pre_computed,
        )

        if return_config_id:
            return game_generator, config_id
        return game_generator

    @staticmethod
    def load_configuration_file(
        config_path: str,
        config_id: int,
        *,
        n_games: int | None = None,
        check_pre_computed: bool = True,
        only_pre_computed: bool = True,
    ) -> Generator[Game, None, None]:
        """Loads a configuration file from the given path.

        Args:
            config_path: The path to the configuration file.
            config_id: The configuration ID to load.

        Returns:
            The loaded configuration as a dictionary.
        """
        params = BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS.copy()
        with open(config_path) as file:
            configurations = json.load(file)
        game_class = GAME_NAME_TO_CLASS_MAPPING[configurations["game_name"]]

        ## Load Configuration ID

        configuration = configurations["configurations"][config_id - 1]
        params.update(configuration)
        ## Load class-specific configurations
        game_should_be_precomputed = configurations["precompute"]
        iteration_param = configurations["iteration_parameter"]
        iteration_param_values = configurations.get(
            "iteration_parameter_values",
            BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS,
        )
        iteration_param_values_names = configurations.get(
            "iteration_parameter_values_names", iteration_param_values
        )
        return GameFactory.create_game_from_configs(
            configuration=configuration,
            game_should_be_precomputed=game_should_be_precomputed,
            iteration_param=iteration_param,
            iteration_param_values=iteration_param_values,
            iteration_param_values_names=iteration_param_values_names,
            game_class=game_class,
            n_players=configurations["n_players"],
            n_games=n_games,
            check_pre_computed=check_pre_computed,
            only_pre_computed=only_pre_computed,
        )


class BenchmarkFactory:
    @staticmethod
    def create_benchmarks_interactive(
        game_config_names: list[str],
        *,
        n_games: list[int] | int = 30,
        approximation_methods: list[str] | None = None,
        index: str = "SV",
        order: int = 1,
        config_path: str = "shapiq-benchmark/configurations_exhaustive/",
        config_save_name: str = "configuration",
    ) -> dict[str, Any]:
        """Create benchmarks from a list of game configuration names.
        Also saves a json file with the configuration details.
        The game_config_names should correspond to the json files in
        shapiq-benchmark/configurations_exhaustive/.

        Args:
            game_config_names (list[str]): List of game configuration names.
            n_games (list[int] | None): List of number of games to create for each configuration.
                If None, all games will be created. Default is None.

        Returns:
            dict: Dictionary of benchmarks with game configuration names as keys and
                game generators as values.
        """
        if approximation_methods is None:
            approximation_methods = [
                "SVARM",
                "KernelSHAP",
                "PermutationSamplingSV",
                "RegressionMSR",
            ]
        benchmarks = {}
        json_dict = {}
        for idx, game_config_name in enumerate(game_config_names):
            game_n_games: int | None = n_games[idx] if isinstance(n_games, list) else n_games
            game_generator, config_id = GameFactory.load_configuration_file_interactive(
                config_path=f"{config_path}{game_config_name}.json",
                n_games=game_n_games,
                check_pre_computed=True,
                only_pre_computed=True,
                return_config_id=True,
            )
            benchmarks[game_config_name + "_" + str(config_id)] = {
                "games": game_generator,
                "approximation_methods": approximation_methods,
                "index": index,
                "order": order,
            }
            json_dict[game_config_name] = {
                "game_config_name": f"{config_path}{game_config_name}.json",
                "n_games": game_n_games,
                "config_id": config_id,
                "approximation_methods": approximation_methods,
                "index": index,
                "order": order,
            }
        # Save json_dict to a json file
        import json

        save_path = Path("shapiq-benchmark/benchmarks/")
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / (config_save_name + ".json"), "w") as f:
            json.dump(json_dict, f, indent=4)
        return benchmarks

    @staticmethod
    def load_benchmarks_from_json(config_path: str):
        import json

        benchmarks = {}
        with open(config_path) as f:
            json_dict = json.load(f)
        for benchmark_name, benchmark_info in json_dict.items():
            game_generator = GameFactory.load_configuration_file(
                config_path=benchmark_info["game_config_name"],
                config_id=benchmark_info["config_id"],
                n_games=benchmark_info["n_games"],
                check_pre_computed=True,
                only_pre_computed=True,
            )
            benchmarks[benchmark_name + "_" + str(benchmark_info["config_id"])] = {
                "games": game_generator,
                "approximation_methods": benchmark_info.get("approximation_methods", []),
                "index": benchmark_info["index"],
                "order": benchmark_info["order"],
            }
        return benchmarks


def load_games_from_configuration(
    game_class: Game | str,
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
    game_class: Game,
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
        game = Game(
            n_players=n_players,
            normalize=False,
            verbose=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"],
        )
        game.load_values(path_to_values)
        return game
    except FileNotFoundError:
        # download the game data if it does not exist
        download_game_data(game_class.get_game_name(), n_players, file_name)
        try:
            game = Game(
                n_players=n_players,
                normalize=False,
                verbose=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"],
            )
            game.load_values(path_to_values)
            return game
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
