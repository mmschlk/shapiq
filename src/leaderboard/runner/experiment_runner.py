"""Experiment runner for the leaderboard."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from leaderboard.metrics.evaluator import compute_all_metrics
from leaderboard.runner.approximator_runner import approximate
from leaderboard.runner.record_builder import create_run_record
from leaderboard.runner.runner_exceptions import InteractionKeyMismatchError, UnknownGameError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from leaderboard.runner.custom_types import InteractionIndex
    from shapiq import Game, InteractionValues
    from shapiq.approximator import Approximator


def align_interaction_values(
    ground_truth: InteractionValues,
    approx_values: InteractionValues,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ground truth and approximated interaction values.

    The function verifies that both "InteractionValues" objects contain the
    same interaction keys and returns two arrays ordered by interaction size and
    interaction tuple.
    The empty interaction () is excluded from the metrics comparison

    Args:
        ground_truth: The exact interaction values.
        approx_values: The approximated interaction values.

    Returns:
        A tuple containing the aligned ground truth values and approximated
        values as numpy arrays.

    Raises:
        InteractionKeyMismatchError: If the interaction keys of both inputs do not match.
    """
    gt_lookup = ground_truth.interaction_lookup
    approx_lookup = approx_values.interaction_lookup

    gt_keys = set(gt_lookup.keys() - {()})
    approx_keys = set(approx_lookup.keys() - {()})

    if gt_keys != approx_keys:
        raise InteractionKeyMismatchError(gt_keys, approx_keys) from None

    interactions = sorted(
        gt_keys,
        key=lambda interaction: (len(interaction), interaction),
    )

    gt_values = np.array(
        [ground_truth.values[gt_lookup[interaction]] for interaction in interactions],
        dtype=float,
    )

    approx_values_aligned = np.array(
        [approx_values.values[approx_lookup[interaction]] for interaction in interactions],
        dtype=float,
    )

    return gt_values, approx_values_aligned


def run_experiment(
    *,
    game: Game,
    game_name: str,
    game_params: dict[str, Any],
    ground_truth: InteractionValues,
    approximator_class: type[Approximator],
    index: InteractionIndex,
    max_order: int,
    budget: int,
    approx_seeds: Iterable[int],
) -> list[dict[str, Any]]:
    """Run approximation experiments for multiple approximation seeds.

    For each seed, the function runs the approximator, aligns the resulting
    interaction values with the ground truth, computes metrics, and stores the
    outcome as a run record. Failed runs are marked and are also stored.

    Args:
        game: The game for which interaction values are approximated.
        game_name: The name of the game.
        game_params: The parameters used to initialize the game.
        ground_truth: The exact interaction values used as reference.
        approximator_class: The approximator class used for the experiment.
        index: The interaction index to approximate.
        max_order: The maximum interaction order to compute.
        budget: The evaluation budget available to the approximator.
        approx_seeds: The approximation seeds to evaluate.

    Returns:
        A list of run records, one for each approximation seed.
    """
    results = []
    for approx_seed in approx_seeds:
        start_time = time.perf_counter()
        try:
            approx_values = approximate(
                game=game,
                approximator_class=approximator_class,
                index=index,
                max_order=max_order,
                budget=budget,
                seed=approx_seed,
            )

            # calculate metrics for each run
            metric_results = compute_all_metrics(
                ground_truth=ground_truth,
                estimated=approx_values,
            )

            runtime_seconds = time.perf_counter() - start_time

            run_record = create_run_record(
                game=game,
                game_name=game_name,
                game_params=game_params,
                approximator_class=approximator_class,
                approximator_params={},
                index=index,
                max_order=max_order,
                budget=budget,
                approx_seed=approx_seed,
                metrics={
                    "mse": metric_results.get("mse"),
                    "mae": metric_results.get("mae"),
                    "mse_normalized": metric_results.get("mse_normalized"),
                    "spearman": metric_results.get("spearman"),
                    "kendall_tau": metric_results.get("kendall_tau"),
                    "precision_at_k": metric_results.get("precision_at_k"),
                },
                runtime_seconds=runtime_seconds,
                run_failed=False,
                error_message=None,
                notes="",
            )

        except (
            NotImplementedError,
            ValueError,
            TypeError,
            RuntimeError,
            InteractionKeyMismatchError,
            UnknownGameError,
        ) as error:
            runtime_seconds = time.perf_counter() - start_time

            run_record = create_run_record(
                game=game,
                game_name=game_name,
                game_params=game_params,
                approximator_class=approximator_class,
                approximator_params={},
                index=index,
                max_order=max_order,
                budget=budget,
                approx_seed=approx_seed,
                metrics=None,
                runtime_seconds=runtime_seconds,
                run_failed=True,
                error_message=str(error),
                notes="",
            )

        results.append(run_record)
    return results
