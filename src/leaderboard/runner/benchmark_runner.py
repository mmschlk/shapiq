"""Benchmark runner for the leaderboard."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from leaderboard.runner.aggregator import aggregate_run_records
from leaderboard.runner.experiment_runner import run_experiment
from leaderboard.runner.ground_truth_computer import compute_ground_truth

if TYPE_CHECKING:
    from collections.abc import Callable

    from leaderboard.runner.custom_types import InteractionIndex
    from shapiq import InteractionValues
    from shapiq.approximator import Approximator
    from shapiq.game import Game

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_benchmark(
    *,
    game: Game,
    game_name: str,
    game_params: dict[str, Any],
    max_order: int,
    approx_seeds: list[int],
    budget: int,
    index: InteractionIndex,
    approximator_class: type[Approximator],
    ground_truth_method: str = "ExactComputer",
    ground_truth_fn: Callable[..., InteractionValues] = compute_ground_truth,
    experiment_fn: Callable[..., list[dict[str, Any]]] = run_experiment,
    aggregate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = aggregate_run_records,
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
        max_order: The maximum interaction order to compute.
        approx_seeds: the concrete approximator seeds for evaluation
        budget: The evaluation budget available to the approximator in each run.
        index: The interaction index to approximate.
        approximator_class: The approximator class used for the benchmark.
        ground_truth_method: The method used to compute the ground truth.
        ground_truth_fn: Function used to compute the ground truth.
        experiment_fn: Function used to run the experiment.
        aggregate_fn: Function used to aggregate the raw records.

    Returns:
        A dictionary containing the raw run records and the aggregated benchmark
        result.

    Raises:
        NoSuccessfulRunsError: Propagated from "aggregate_run_records" when no successful
            run records are available for aggregation.
    """
    # Define the values

    # Compute ground truth
    ground_truth = ground_truth_fn(
        game=game, index=index, max_order=max_order, method=ground_truth_method
    )

    # approximate values [n times]
    results = experiment_fn(
        game=game,
        game_name=game_name,
        game_params=game_params,
        ground_truth=ground_truth,
        approximator_class=approximator_class,
        index=index,
        max_order=max_order,
        budget=budget,
        approx_seeds=approx_seeds,
    )

    # Override the hardcoded 'ExactComputer' in raw records with the actual method used
    for record in results:
        record["ground_truth_method"] = ground_truth_method

        # debugging
    for record in results:
        logger.debug("failed: %s", record["run_failed"])
        logger.debug("error: %s\n", record["error_message"])

        # aggregation
    aggregated_result = aggregate_fn(results)

    # print-out
    logger.info("number of raw results: %d", len(results))
    logger.info("First raw run record:")
    logger.info(json.dumps(results[0], indent=2))
    logger.info("\nAggregated result:")
    logger.info(json.dumps(aggregated_result, indent=2))

    return {
        "raw_results": results,
        "aggregated_result": aggregated_result,
    }
