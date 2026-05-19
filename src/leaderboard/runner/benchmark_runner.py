"""Benchmark runner for the leaderboard."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from leaderboard.runner.aggregator import aggregate_run_records
from leaderboard.runner.experiment_runner import run_experiment
from leaderboard.runner.ground_truth_computer import compute_ground_truth

if TYPE_CHECKING:
    from leaderboard.runner.custom_types import InteractionIndex
    from shapiq.approximator import Approximator
    from shapiq.game import Game

import logging

logging.basicConfig(level=logging.INFO)


def run_benchmark(
    *,
    game: Game,
    game_name: str,
    game_params: dict[str, Any],
    game_seed: int,
    max_order: int,
    number_of_different_approx_seeds: int,
    budget: int,
    index: InteractionIndex,
    approximator_class: type[Approximator],
) -> dict[str, Any]:
    """Run a complete benchmark for one defined setup.

    The benchmark computes the ground truth interaction values, runs the given
    approximator with multiple approximation seeds, and aggregates the resulting
    run records. It then returns the raw records and the aggregated record
    in a single summary record.

    Args:
        game: The game for which interaction values are approximated.
        game_name: The name of the game.
        game_params: The parameters used to initialize the game.
        game_seed: The random seed used to initialize the game.
        max_order: The maximum interaction order to compute.
        number_of_different_approx_seeds: The number of approximation seeds to
            evaluate.
        budget: The evaluation budget available to the approximator in each run.
        index: The interaction index to approximate.
        approximator_class: The approximator class used for the benchmark.

    Returns:
        A dictionary containing the raw run records and the aggregated benchmark
        result.

    Raises:
        ValueError: Propagated from "aggregate_run_records" when no successful
            run records are available for aggregation.
    """
    # Define the values
    # TO DO: Index approximator validation (e.g. certain indices like SV expect specific order(1)! )
    approx_seeds = range(number_of_different_approx_seeds)

    # Compute ground truth
    ground_truth = compute_ground_truth(game=game, index=index, max_order=max_order)

    # approximate values [n times]
    results = run_experiment(
        game=game,
        game_name=game_name,
        game_params=game_params,
        game_seed=game_seed,
        ground_truth=ground_truth,
        approximator_class=approximator_class,
        index=index,
        max_order=max_order,
        budget=budget,
        approx_seeds=approx_seeds,
    )

    # debugging
    for record in results:
        logging.debug("failed: %s", record["run_failed"])
        logging.debug("error: %s\n", record["error_message"])

    # aggregation
    aggregated_result = aggregate_run_records(results)

    # print-out
    logging.info("number of raw results: %d", len(results))
    logging.info("First raw run record:")
    logging.info(json.dumps(results[0], indent=2))
    logging.info("\nAggregated result:")
    logging.info(json.dumps(aggregated_result, indent=2))

    return {
        "raw_results": results,
        "aggregated_result": aggregated_result,
    }
