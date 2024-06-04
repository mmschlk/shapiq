"""This module contains the main benchmark run setup for the shapiq package."""

import copy
import multiprocessing as mp
import os
import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ...approximator._base import Approximator
from ...interaction_values import InteractionValues
from ..base import Game
from .metrics import get_all_metrics

BENCHMARK_RESULTS_DIR = "results"


def save_results(results: pd.DataFrame, save_path: str) -> None:
    """Save the results of the benchmark as a CSV file.

    Args:
        results: The results of the benchmark.
        save_path: The path to save the results as a CSV file. Defaults to "results.csv".
    """
    # check if the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir != "" and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results.to_json(save_path)


def run_benchmark(
    index: str,
    order: int,
    games: list[Game],
    gt_values: Optional[list[InteractionValues]] = None,
    approximators: Optional[
        Union[list[Approximator], list[Approximator.__class__], list[str]]
    ] = None,
    budget_steps: Optional[list[Union[int]]] = None,
    budget_step: float = 0.05,
    max_budget: Optional[int] = None,
    n_iterations: int = 1,
    n_jobs: int = 1,
    benchmark_name: str = "benchmark",
    save: bool = True,
    save_path: Optional[str] = None,
    rerun_if_exists: bool = False,
) -> pd.DataFrame:
    """Run the benchmark for the given approximators and games.

    Args:
        index: The index to run the benchmark on (e.g., "SV", "k-SII").
        order: The order of the index to run the benchmark on (e.g., 1, 2).
        approximators: The list of approximators to benchmark.
        games: The list of games to benchmark. The games should have the same number of players.
        gt_values: The list of ground truth values for the games. The length of the list should be
            the same as the number of games.
        budget_steps: The list of budget steps to benchmark on. Can be either a list of integers or
            floats. If a float is provided, the budget is calculated as the percentage of the
            maximum budget for the game `(2**n_players)`.
        budget_step: The step size where the approximators are evaluated (in percentage of the
            maximum budget). Defaults to 0.05.
        max_budget: The maximum budget to evaluate the approximators on. Defaults to the maximum
            budget for the game `(2**n_players)`.
        n_iterations: The number of iterations to run the benchmark for. Each iteration runs all
            approximators on all games for all budget steps.
        n_jobs: The number of parallel jobs to run. Defaults to 1.
        benchmark_name: The name of the benchmark. Defaults to "benchmark".
        save: If `True`, the results are saved as a JSON file. Defaults to `True`.
        save_path: The path to save the results as a JSON file. Defaults to
            `f"results/{benchmark_name}.json`.
        rerun_if_exists: If `True`, the benchmark is rerun even if the results already exist.
            Defaults to `False`.

    Returns:
        The results of the benchmark.

    Raises:
        ValueError: If the number of players in the games is not the same.
        ValueError: If the number of ground truth values is not the same as the number of games.
    """
    from .benchmark_config import APPROXIMATION_CONFIGURATIONS, APPROXIMATION_NAME_TO_CLASS_MAPPING

    if save_path is None:
        save_path = os.path.join("results", f"{benchmark_name}.json")

    if not rerun_if_exists and os.path.exists(save_path):
        print(f"Results for the benchmark {benchmark_name} already exist. Skipping the benchmark.")
        return pd.read_json(save_path)

    # check that all games have the same number of players
    n_players = games[0].n_players
    if not all(game.n_players == n_players for game in games):
        raise ValueError("All games must have the same number of players.")

    # check that the number of ground truth values is the same as the number of games
    if gt_values is None:
        print("Computing the exact values for the games.")
        gt_values = []
        for game in tqdm(games, unit=" games"):
            gt_values.append(game.exact_values(index=index, order=order))

    if len(gt_values) != len(games):
        raise ValueError(
            "The number of ground truth values must be the same as the number of games."
        )

    # transform the budget steps to integers if float is provided
    if n_players > 16:  # sets the budget to 10k for synthetic games with more than 16 players
        max_budget = 10_000
    max_budget = max_budget or 2**n_players  # set budget to 2**n_players if not provided
    if budget_steps is None:
        budget_steps = [
            int(round(budget_step, 2) * max_budget)
            for budget_step in np.arange(budget_step, 1.0 + budget_step + budget_step, budget_step)
        ]
        print("Created budget steps: ", budget_steps)

    # get approximators
    if approximators is None:
        approximators = APPROXIMATION_CONFIGURATIONS[index]
    # get approx classes if strings are provided
    approximators = [
        APPROXIMATION_NAME_TO_CLASS_MAPPING[approx] if isinstance(approx, str) else approx
        for approx in approximators
    ]
    approximators_per_iteration = {}
    for iteration in range(1, n_iterations + 1):
        approximators_per_iteration[iteration] = [
            (
                _init_approximator_from_class(
                    approximator_class, n_players, index, order, random_state=iteration
                )
                if isinstance(approximator_class, type)  # check if the approximator is a class
                else approximator_class
            )
            for approximator_class in approximators
        ]
    print(f"Got {len(approximators)} approximators for the benchmark.")
    print(f"All approximators: {approximators_per_iteration}")

    # create the parameter space for the benchmark
    parameter_space = [
        (iteration, approximator, game, gt_value, budget_step)
        for iteration in range(1, n_iterations + 1)
        for approximator in approximators_per_iteration[iteration]
        for game, gt_value in zip(games, gt_values)
        for budget_step in budget_steps
    ]

    # shuffle the parameter space for better estimation of the time
    new_indices = np.random.permutation(len(parameter_space))
    parameter_space_shuffled = [parameter_space[i] for i in new_indices]
    parameter_space = parameter_space_shuffled

    # run the benchmark
    if n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_run_benchmark, parameter_space),
                    total=len(parameter_space),
                    desc="Running benchmark:",
                    unit=" experiments",
                )
            )
    else:
        progress = tqdm(
            total=len(approximators) * len(games) * sum(budget_steps) * n_iterations,
            desc="Running benchmark:",
            unit=" game evaluations",
        )
        results = []
        for args in parameter_space:
            results.append(_run_benchmark(args))
            budget_step = args[-1]
            progress.update(budget_step)
        progress.close()

    # add the exact values to the results
    for game, gt_value in zip(games, gt_values):
        results.append(
            {
                "game_name": game.game_name,
                "game_id": game.game_id,
                "n_players": game.n_players,
                "budget": 2**game.n_players,
                "budget_relative": 1.0,
                "approximator": "Exact",
                "estimates_values": gt_value.dict_values,
                "used_budget": 2**game.n_players,
                "estimated": False,
            }
        )

    # finalize results
    results_df = pd.DataFrame(results)
    if save:  # save the results as a json file
        save_results(results_df, save_path=save_path)
    return results_df


def _run_benchmark(args) -> dict[str, Union[str, int, float, InteractionValues]]:
    """Run the benchmark for a given set of parameters.

    Args:
        args: The parameters for the benchmark as a tuple of `(iteration, approximator, game,
            gt_value, budget)`.

    Returns:
        The results of the benchmark.
    """
    iteration, approximator, game, gt_value, budget = args
    estimates = copy.deepcopy(approximator.approximate(budget=budget, game=game))
    result = {
        "iteration": iteration,
        "game_name": game.game_name,
        "game_id": game.game_id,
        "n_players": game.n_players,
        "budget": budget,
        "budget_relative": round(budget / (2**game.n_players), 6),
        "approximator": approximator.__class__.__name__,
        "estimates_values": estimates.dict_values,
        "used_budget": estimates.estimation_budget,
        "estimated": estimates.estimated,
    }
    # compute the metrics
    metrics_all_orders = get_all_metrics(gt_value, estimates)
    result.update(metrics_all_orders)
    for order in range(1, gt_value.max_order + 1):
        metrics_order = get_all_metrics(
            gt_value.get_n_order(order), estimates.get_n_order(order), order_indicator=str(order)
        )
        result.update(metrics_order)
    return result


def _init_approximator_from_class(
    approximator_class: Approximator.__class__,
    n_players: int,
    index: str,
    max_order: int,
    random_state: int,
) -> Approximator:
    """Initializes an approximator from a class.

    Args:
        approximator_class: The approximator class to initialize.
        n_players: The number of players.
        index: The index to initialize the approximator with.
        max_order: The maximum order to initialize the approximator with.

    Returns:
        The initialized approximator.
    """
    from .benchmark_config import APPROXIMATION_BENCHMARK_PARAMS

    params_tuple = APPROXIMATION_BENCHMARK_PARAMS[approximator_class]
    all_par = {"index": index, "max_order": max_order, "random_state": random_state, "n": n_players}
    init_param = {}
    for param in params_tuple:
        init_param[param] = all_par[param]
    return approximator_class(**init_param)


def load_benchmark_results(
    save_path: Optional[str] = None,
    index: Optional[str] = None,
    order: Optional[int] = None,
    game_class: Optional[Union[Game.__class__, str]] = None,
    game_configuration: Optional[Union[dict[str, Any], int]] = None,
    game_n_player_id: Optional[int] = None,
    game_n_games: Optional[int] = None,
) -> tuple[pd.DataFrame, str]:
    """Loads the benchmark results from a JSON file which either specified by the save path or
    the benchmark configuration.

    Args:
        save_path: The path to the JSON file to load the results from. Defaults to None.
        index: The index to load the results for. Defaults to None.
        order: The order to load the results for. Defaults to None.
        game_class: The game class to load the results for. Defaults to None.
        game_configuration: The configuration to load the results for. Defaults to None.
        game_n_player_id: The player ID to load the results for. Defaults to None.
        game_n_games: The number of games to load the results for. Defaults to None.

    Returns:
        The loaded benchmark results as a pandas DataFrame.

    Raises:
        ValueError: If save path is None and the game configuration is not provided.
    """
    if save_path is None:
        from .benchmark_config import (
            BENCHMARK_CONFIGURATIONS,
            get_game_class_from_name,
            get_game_file_name_from_config,
        )

        if game_class is None or game_configuration is None or game_n_player_id is None:
            raise ValueError("The game configuration must be provided if the save path is not.")

        if isinstance(game_class, str):
            game_class = get_game_class_from_name(game_class)

        if isinstance(game_configuration, int):
            game_configuration = BENCHMARK_CONFIGURATIONS[game_class][game_n_player_id][
                "configurations"
            ][game_configuration - 1]

        save_path = os.path.join(
            BENCHMARK_RESULTS_DIR,
            _make_benchmark_name(
                config_id=get_game_file_name_from_config(game_configuration),
                game_class=game_class,
                n_games=game_n_games,
                index=index,
                order=order,
            )
            + ".json",
        )

    df = pd.read_json(save_path)
    return df, save_path


def run_benchmark_from_configuration(
    index: str,
    order: int,
    *,
    game_class: Union[Game.__class__, str],
    game_configuration: Union[dict[str, Any], int] = 1,
    game_n_player_id: int = 0,
    game_n_games: Optional[int] = None,
    approximators: Optional[
        Union[list[Approximator], list[Approximator.__class__], list[str]]
    ] = None,
    max_budget: Optional[int] = None,
    n_iterations: int = 1,
    n_jobs: int = 1,
    rerun_if_exists: bool = False,
) -> None:
    """Runs a benchmark on a specified configuration.

    Args:
        index: The index to run the benchmark on (e.g., "SV", "k-SII").
        order: The order of the index to run the benchmark on (e.g., 1, 2).
        game_class: The game class to run the benchmark on.
        game_configuration: The configuration to run the benchmark on or the configuration ID to run the
            benchmark on. Defaults to 1 (first configuration as specified in
            `benchmark_config.BENCHMARK_CONFIGURATIONS`).
        game_n_player_id: The player ID to use for the benchmark. Defaults to 0.
        game_n_games: The number of games to load for the benchmark. If None, all games are loaded.
            Defaults to None.
        approximators: The list of approximators to benchmark. If None, all approximators are used
            that can be used for the given index. Defaults to None.
        max_budget: The maximum budget to evaluate the approximators on. Defaults to the maximum
            budget for the game `(2**n_players)`.
        n_iterations: The number of iterations to run the benchmark for. Each iteration runs all
            approximators on all games for all budget steps. Defaults to 1.
        n_jobs: The number of parallel jobs to run. Defaults to 1.
        rerun_if_exists: If `True`, the benchmark is rerun even if the results already exist.
            Defaults to `False`.
    """
    from .benchmark_config import (
        BENCHMARK_CONFIGURATIONS,
        get_game_class_from_name,
        get_game_file_name_from_config,
        get_name_from_game_class,
        load_games_from_configuration,
    )

    game_class = get_game_class_from_name(game_class) if isinstance(game_class, str) else game_class
    game_name = get_name_from_game_class(game_class)
    print(f"Running benchmark for the game: {game_name}.")

    # get configuration from the benchmark configurations
    if isinstance(game_configuration, int):
        game_configuration = BENCHMARK_CONFIGURATIONS[game_class][game_n_player_id][
            "configurations"
        ][game_configuration - 1]

    # load the games
    config_id = get_game_file_name_from_config(game_configuration)
    games: list[Game] = list(
        load_games_from_configuration(
            game_class, game_configuration, n_player_id=game_n_player_id, only_pre_computed=True
        )
    )
    if game_n_games is not None:
        games = games[:game_n_games]
    print(f"Loaded {len(games)} games for the benchmark. Configuration ID: {config_id}.")
    if not all(game.precomputed for game in games):
        warnings.warn("Not all games are pre-computed. The benchmark might take longer to run.")
    if not all(game.is_normalized for game in games):
        warnings.warn("Not all games are normalized. The benchmark might not be accurate.")

    # get the benchmark name for saving the results
    benchmark_name = _make_benchmark_name(config_id, game_class, len(games), index, order)
    save_path = os.path.join("results", f"{benchmark_name}.json")
    print(
        f"Checking if the benchmark results already exist with the name: {benchmark_name} and the "
        f"save path: {save_path}."
    )
    if not rerun_if_exists and os.path.exists(save_path):
        print(f"Results for the benchmark {benchmark_name} already exist. Skipping the benchmark.")
        return
    elif rerun_if_exists:
        print(f"Rerunning the benchmark {benchmark_name}.")
    else:
        print(f"Results for the benchmark {benchmark_name} do not exist. Running the benchmark.")

    # get the exact values
    print("Computing the exact values for the games.")
    gt_values = [game.exact_values(index=index, order=order) for game in tqdm(games, unit=" games")]

    # run the benchmark
    run_benchmark(
        index=index,
        order=order,
        approximators=approximators,
        games=games,
        gt_values=gt_values,
        benchmark_name=benchmark_name,
        n_jobs=n_jobs,
        max_budget=max_budget,
        n_iterations=n_iterations,
        save=True,
        rerun_if_exists=rerun_if_exists,
        save_path=save_path,
    )


def _make_benchmark_name(
    config_id: str, game_class: Union[Game.__class__, str], n_games: int, index: str, order: int
) -> str:
    """Make the benchmark name for the given configuration."""
    try:
        game_name = game_class.get_game_name()
    except AttributeError:  # game_class is a string
        game_name = game_class
    return "_".join(
        [
            game_name,
            str(config_id),
            str(index),
            str(order),
            f"n_games={n_games}",
        ]
    )


# Path: shapiq/games/benchmark/run.py
