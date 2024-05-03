"""This module contains the main benchmark run setup for the shapiq package."""

import copy
import multiprocessing as mp
import os
from typing import Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from ...approximator._base import Approximator
from ...interaction_values import InteractionValues
from ..base import Game
from .metrics import get_all_metrics


def save_results(results, save_path: str = "results.csv") -> None:
    """Save the results of the benchmark as a CSV file.

    Args:
        results: The results of the benchmark.
        save_path: The path to save the results as a CSV file. Defaults to "results.csv".
    """
    # check if the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir != "" and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.DataFrame(results)
    if "estimates" in df.columns:  # drop the estimates column if it exists
        df = df.drop(columns=["estimates"])
    df.to_csv(save_path, index=False)


def run_benchmark(
    approximators: list[Approximator],
    games: list[Game],
    gt_values: list[InteractionValues],
    budget_steps: list[Union[int, float]],
    n_iterations: int = 1,
    n_jobs: int = 1,
    save_path: Optional[str] = "results.csv",
) -> list[dict[str, Union[str, int, float, InteractionValues]]]:
    """Run the benchmark for the given approximators and games.

    Args:
        approximators: The list of approximators to benchmark.
        games: The list of games to benchmark. The games should have the same number of players.
        gt_values: The list of ground truth values for the games. The length of the list should be
            the same as the number of games.
        budget_steps: The list of budget steps to benchmark on. Can be either a list of integers or
            floats. If a float is provided, the budget is calculated as the percentage of the
            maximum budget for the game `(2**n_players)`.
        n_iterations: The number of iterations to run the benchmark for. Each iteration runs all
            approximators on all games for all budget steps.
        n_jobs: The number of parallel jobs to run. Defaults to 1.
        save_path: The path to save the results as a CSV file. Defaults to "results.csv".

    Returns:
        The results of the benchmark.

    Raises:
        ValueError: If the number of players in the games is not the same.
        ValueError: If the number of ground truth values is not the same as the number of games.
    """
    # check that all games have the same number of players
    n_players = games[0].n_players
    if not all(game.n_players == n_players for game in games):
        raise ValueError("All games must have the same number of players.")

    # check that the number of ground truth values is the same as the number of games
    if len(gt_values) != len(games):
        raise ValueError(
            "The number of ground truth values must be the same as the number of games."
        )

    # transform the budget steps to integers if float is provided
    budget_steps = [
        int(budget_step * 2**n_players) if isinstance(budget_step, float) else budget_step
        for budget_step in budget_steps
    ]

    # create the parameter space for the benchmark
    parameter_space = [
        (iteration, approximator, game, gt_value, budget_step)
        for iteration in range(1, n_iterations + 1)
        for approximator in approximators
        for game, gt_value in zip(games, gt_values)
        for budget_step in budget_steps
    ]

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

    # save the results as a CSV file
    if save_path is not None:
        save_results(results, save_path=save_path)

    return results


def _run_benchmark(
    args: tuple[int, Approximator, Game, InteractionValues, int]
) -> dict[str, Union[str, int, float, InteractionValues]]:
    """Run the benchmark for a given set of parameters.

    Args:
        args: The tuple of parameters for the benchmark.

    Returns:
        The results of the benchmark.
    """
    iteration, approximator, game, gt_value, budget = args
    estimates = copy.deepcopy(approximator.approximate(budget=budget, game=game))
    result = {
        "iteration": iteration,
        "game_name": game.get_game_name,
        "game_id": game.game_id,
        "n_players": game.n_players,
        "budget": budget,
        "budget_relative": round(budget / (2**game.n_players), 2),
        "approximator": approximator.__class__.__name__,
        "approximator_id": approximator.approximator_id,
        "estimates": estimates,
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
