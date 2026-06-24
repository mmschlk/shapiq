"""Experiment runner for the leaderboard."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from shapiq.approximator.sparse import Sparse

from leaderboard.metrics.evaluator import compute_all_metrics
from leaderboard.runner.approximator_runner import approximate
from leaderboard.runner.record_builder import create_run_record
from leaderboard.runner.runner_exceptions import InteractionKeyMismatchError, UnknownGameError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from leaderboard.runner.custom_types import InteractionIndex
    from shapiq import Game, InteractionValues
    from shapiq.approximator import Approximator


def align_interaction_values(
    ground_truth: InteractionValues,
    approx_values: InteractionValues,
    *,
    default_approx_value: float = 0.0,
    allow_missing_approx_keys: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ground truth and approximated interaction values.

    The ground truth defines the evaluation space. The returned arrays are ordered
    by interaction size and interaction tuple. The empty interaction ``()`` is
    excluded from the metrics comparison.

    Missing approximated interactions are filled with ``default_approx_value``.
    This supports sparse approximators that intentionally return only a subset of
    interactions. Additional approximated interactions that are not present in the
    ground truth are treated as an alignment error.

    Args:
        ground_truth: The exact interaction values.
        approx_values: The approximated interaction values.
        default_approx_value: value used for ground-truth interactions that are
            missing from the approximation. Defaults to ``0.0``.
        allow_missing_approx_keys: Defines the policy regarding missing approximation keys.

    Returns:
        A tuple containing the aligned ground truth values and approximated
        values as numpy arrays.

    Raises:
        InteractionKeyMismatchError: If approximated values contain interaction
        keys that are not present in the ground truth, or if approximated values
        are missing ground-truth keys while ``allow_missing_approx_keys`` is false.
    """
    gt_lookup = ground_truth.interaction_lookup
    approx_lookup = approx_values.interaction_lookup

    gt_keys = set(gt_lookup.keys()) - {()}
    approx_keys = set(approx_lookup.keys()) - {()}

    missing_keys = gt_keys - approx_keys
    additional_approx_keys = approx_keys - gt_keys

    if missing_keys and not allow_missing_approx_keys:
        raise InteractionKeyMismatchError(gt_keys, approx_keys) from None

    if additional_approx_keys:
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
        [
            approx_values.values[approx_lookup[interaction]]
            if interaction in approx_lookup
            else default_approx_value
            for interaction in interactions
        ],
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
        run_record = run_single_experiment_seed(
            game=game,
            game_name=game_name,
            game_params=game_params,
            ground_truth=ground_truth,
            approximator_class=approximator_class,
            index=index,
            max_order=max_order,
            budget=budget,
            approx_seed=approx_seed,
        )
        results.append(run_record)
    return results


def run_single_experiment_seed(
    *,
    game: Game,
    game_name: str,
    game_params: dict[str, Any],
    ground_truth: InteractionValues,
    approximator_class: type[Approximator],
    index: InteractionIndex,
    max_order: int,
    budget: int,
    approx_seed: int,
    approximate_fn: Callable[..., InteractionValues] = approximate,
    align_fn: Callable[..., tuple[np.ndarray, np.ndarray]] = align_interaction_values,
    metrics_fn: Callable[..., dict[str, Any]] = compute_all_metrics,
    record_builder_fn: Callable[..., dict[str, Any]] = create_run_record,
) -> dict[str, Any]:
    """Run one approximation experiment for a single approximation seed.

    The function executes the approximator for one seed, aligns the resulting
    interaction values with the provided ground truth, computes evaluation
    metrics, and builds a raw run record. If one of the expected runner-level
    errors occurs, the function returns a failed run record instead of raising
    the error.

    The optional function parameters are dependency-injection hooks. They
    default to the production implementations, but can be replaced in tests to
    avoid running real approximators, metrics, or record-building logic.

    Args:
        game: The game for which interaction values are approximated.
        game_name: The name of the game.
        game_params: The parameters used to initialize the game.
        ground_truth: The exact interaction values used as reference.
        approximator_class: The approximator class used for the experiment.
        index: The interaction index to approximate.
        max_order: The maximum interaction order to compute.
        budget: The evaluation budget available to the approximator.
        approx_seed: The approximation seed used for this single run.
        approximate_fn: Function used to compute approximated interaction values.
        align_fn: Function used to align ground truth and approximated values.
        metrics_fn: Function used to compute metrics from aligned values.
        record_builder_fn: Function used to create the raw run record.

    Returns:
        A raw run record dictionary for the given approximation seed. The record
        is marked as failed if an expected approximation, alignment, metric, or
        record-building error occurred.
    """
    start_time = time.perf_counter()
    try:
        approx_values = approximate_fn(
            game=game,
            approximator_class=approximator_class,
            index=index,
            max_order=max_order,
            budget=budget,
            seed=approx_seed,
        )

        # align interaction values
        allow_missing_approx_keys = _allows_sparse_interaction_output(approximator_class)

        gt_values_aligned, approx_values_aligned = align_fn(
            ground_truth,
            approx_values,
            allow_missing_approx_keys=allow_missing_approx_keys,
        )

        # calculate metrics for each run
        metric_results = metrics_fn(
            ground_truth=gt_values_aligned,
            estimated=approx_values_aligned,
        )

        runtime_seconds = time.perf_counter() - start_time

        run_record = record_builder_fn(
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

        run_record = record_builder_fn(
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
    return run_record

def _allows_sparse_interaction_output(
    approximator_class: type[Approximator],
) -> bool:
    """Return whether an approximator may intentionally return sparse output."""
    return issubclass(approximator_class, Sparse)
