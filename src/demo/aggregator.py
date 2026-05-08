import numpy as np
import uuid
from datetime import datetime, timezone


def aggregate_run_records(run_records: list[dict]) -> dict:
    successful_runs = []
    mse_values_list = []
    mae_values_list = []
    runtime_values = []

    for record in run_records:
        if record["run_failed"]:
            continue

        successful_runs.append(record)

        mse_value = record["metrics"]["mse"]
        if mse_value is not None:
            mse_values_list.append(mse_value)

        mae_value = record["metrics"]["mae"]
        if mae_value is not None:
            mae_values_list.append(mae_value)

        runtime_value = record["runtime_seconds"]
        if runtime_value is not None:
            runtime_values.append(runtime_value)

    if not successful_runs:
        raise ValueError("No successful runs to aggregate.")

    first_record = successful_runs[0]

    mse_values = np.array(mse_values_list)
    mae_values = np.array(mae_values_list)

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

        "metrics": {
            "mse": float(np.mean(mse_values)) if len(mse_values) > 0 else None,
            "mae": float(np.mean(mae_values)) if len(mae_values) > 0 else None,
            "mse_normalized": None,
            "spearman": None,
            "kendall_tau": None,
            "precision_at_k": None,
        },

        "runtime_seconds": runtime_seconds,

        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": first_record["hardware"],
        "notes": "",
    }