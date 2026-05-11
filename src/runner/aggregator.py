import numpy as np
import uuid
from datetime import datetime, timezone


METRIC_KEYS = [
    "mse",
    "mae",
    "mse_normalized",
    "spearman",
    "kendall_tau",
    "precision_at_k",
]


def aggregate_metric_values(successful_runs: list[dict]) -> dict:
    aggregated_metrics = {}

    for metric_name in METRIC_KEYS:
        values = []

        for record in successful_runs:
            if "metrics" not in record:
                raise KeyError("Successful run record is missing 'metrics'.")

            if record["metrics"] is None:
                raise ValueError("Successful run record has metrics=None.")

            metrics = record["metrics"]
            value = metrics.get(metric_name)

            if value is not None:
                values.append(value)

        if values:
            aggregated_metrics[metric_name] = float(np.mean(values))
        else:
            aggregated_metrics[metric_name] = None

    return aggregated_metrics


def aggregate_run_records(run_records: list[dict]) -> dict:
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
        raise ValueError("No successful runs to aggregate.")

    first_record = successful_runs[0]

    if runtime_values:
        runtime_seconds = float(np.mean(np.array(runtime_values)))
    else:
        runtime_seconds = None

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

        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": first_record["hardware"],
        "notes": "",
    }