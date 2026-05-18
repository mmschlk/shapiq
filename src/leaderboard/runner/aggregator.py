"""Aggregator Module for the ShapIQ Living Benchmark Leaderboard.

This module provides functionality to aggregate multiple run records into a single representative record.
The aggregation process includes:
- Averaging metric values across successful runs;
- Averaging runtime values across successful runs;
- Retaining constant values from the first successful run (e.g., game parameters, hardware information);
- Generating a new unique run ID and timestamp for the aggregated record.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .runner_exceptions import MissingMetricsKeyError, NoSuccessfulRunsError, NullMetricsError

METRIC_KEYS = [
    "mse",
    "mae",
    "mse_normalized",
    "spearman",
    "kendall_tau",
    "precision_at_k",
]


def aggregate_metric_values(successful_runs: list[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregates metric values for all metrics across all successful runs.

    Args:
        successful_runs: The list of runs

    Returns:
        A dictionary mapping each metric name to its mean value across all successful runs.
        Metrics without values are mapped to ``None``.

    Raises:
        KeyError: If a metrics entry is missing in the run.
        ValueError: If metrics = None.
    """
    aggregated_metrics = {}

    for metric_name in METRIC_KEYS:
        values = []

        for record in successful_runs:
            if "metrics" not in record:
                raise MissingMetricsKeyError from None

            if record["metrics"] is None:
                raise NullMetricsError from None

            metrics = record["metrics"]
            value = metrics.get(metric_name)

            if value is not None:
                values.append(value)

        if values:
            aggregated_metrics[metric_name] = float(np.mean(values))
        else:
            aggregated_metrics[metric_name] = None

    return aggregated_metrics


def aggregate_run_records(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates a list of run records into a single record.

    Args:
        run_records: the list of runs

    Returns:
        A single run_record with aggregated values.
        For values that are constant across runs the first instance is taken.
        The resulting run also represents runtime and hardware information

    Raises:
        ValueError: If no successful run records are available for aggregation.
    """
    successful_runs = []
    runtime_values = []

    for record in run_records:
        if record["run_failed"]:
            continue

        successful_runs.append(record)

        runtime_value = record.get("runtime_seconds")
        if runtime_value is not None:
            runtime_values.append(runtime_value)

    if not successful_runs:
        raise NoSuccessfulRunsError from None

    first_record = successful_runs[0]

    runtime_seconds = float(np.mean(np.array(runtime_values))) if runtime_values else None

    return {
        "run_id": str(uuid.uuid4()),
        "game_name": first_record["game_name"],
        "game_id": first_record["game_id"],
        "game_params": first_record["game_params"],
        "n_players": first_record["n_players"],
        "approximator_name": first_record["approximator_name"],
        "approximator_params": first_record["approximator_params"],
        "shapiq_version": first_record["shapiq_version"],
        "index": first_record["index"],
        "max_order": first_record["max_order"],
        "budget": first_record["budget"],
        "approx_seed": None,
        "ground_truth_method": first_record["ground_truth_method"],
        "run_failed": False,
        "error_message": None,
        "metrics": aggregate_metric_values(successful_runs),
        "runtime_seconds": runtime_seconds,
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": first_record["hardware"],
        "notes": "",
    }
