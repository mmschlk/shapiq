from shapiq_games.synthetic import SOUM
from shapiq.approximator import ProxySHAP, Approximator
from custom_types import InteractionIndex, MetricFunction
from metrics import mse_metric, mae_metric
from runner import approximate, compute_ground_truth, compute_metrics
import platform
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from shapiq.game import Game
import numpy as np
import json

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


#TODO: Placeholder-implementation
def get_hardware_info() -> dict:
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": None,
        "python_version": platform.python_version(),
    }


def create_run_record(
    *,
    game: Game,
    game_name: str,
    game_params: dict,
    approximator_class: type[Approximator],
    approximator_params: dict,
    index: str,
    max_order: int,
    budget: int,
    approx_seed: int,
    metrics: dict[str, float] | None,
    runtime_seconds: float | None,
    run_failed: bool,
    error_message: str | None,
    notes: str = "",
) -> dict:
    return {
        "run_id": str(uuid.uuid4()), #TODO: needs to be checked

        "game_name": game_name,
        "game_id": game.game_id,
        "game_params": game_params,
        "n_players": game.n_players,

        "approximator_name": approximator_class.__name__,
        "approximator_params": approximator_params,
        "shapiq_version": version("shapiq"),

        "index": index,
        "max_order": max_order,
        "budget": budget,
        "approx_seed": approx_seed,

        "ground_truth_method": "ExactComputer", #TODO: this needs to be determined during the process

        "run_failed": run_failed,
        "error_message": error_message,

        "metrics": metrics if metrics is not None else {
            "mse": None,
            "mae": None,
            "mse_normalized": None,
            "spearman": None,
            "kendall_tau": None,
            "precision_at_k": None,
        },

        "runtime_seconds": runtime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": get_hardware_info(),
        "notes": notes,
    }


def demo() -> None:
    print("This is a test to see how game approximation works")

    # Define the values
    game_seed = 42
    max_order = 2
    number_of_different_approx_seeds = 5
    approx_seeds = range(number_of_different_approx_seeds)
    # select index (certain indices like SV expect specific order(1)! )
    # We probably also need to check which approximator supports which index.
    index : InteractionIndex = "SII"
    game = SOUM(n=10, n_basis_games=20, min_interaction_size=1,
                max_interaction_size=max_order, random_state=game_seed)
    #We only provide the approximator class to build approximators with different seeds
    approximator_class: type[Approximator] = ProxySHAP
    budget = 100
    #metrics are just functions for now, this should be refined into a clean easy to work with structure
    #maybe build a metric object
    metrics: dict[str, MetricFunction] = {
        "mse": mse_metric,
        "mae": mae_metric,
    }


    #Compute ground truth
    ground_truth = compute_ground_truth(game=game, index=index, max_order=max_order)

    # approximate values [n times]
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

            #calculate metrics for each run
            metric_results: dict[str, float] = compute_metrics(
                ground_truth=ground_truth,
                approximation=approx_values,
                metrics=metrics
            )

            runtime_seconds = time.perf_counter() - start_time

            run_record = create_run_record(
                game=game,
                game_name="SOUM",
                game_params={
                    "n_basis_games": 20,
                    "min_interaction_size": 1,
                    "max_interaction_size": max_order,
                    "random_state": game_seed,
                },
                approximator_class=approximator_class,
                approximator_params={
                    "random_state": approx_seed,
                },
                index=index,
                max_order=max_order,
                budget=budget,
                approx_seed=approx_seed,
                metrics={
                    "mse": metric_results.get("mse"),
                    "mae": metric_results.get("mae"),
                    "mse_normalized": None,
                    "spearman": None,
                    "kendall_tau": None,
                    "precision_at_k": None,
                },
                runtime_seconds=runtime_seconds,
                run_failed=False,
                error_message=None,
                notes="",
            )

        except Exception as error:
            runtime_seconds = time.perf_counter() - start_time

            run_record = create_run_record(
                game=game,
                game_name="SOUM",
                game_params={
                    "n_basis_games": 20,
                    "min_interaction_size": 1,
                    "max_interaction_size": max_order,
                    "random_state": game_seed,
                },
                approximator_class=approximator_class,
                approximator_params={
                    "random_state": approx_seed,
                },
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

    # aggregation
    aggregated_result = aggregate_run_records(results)
    print("First raw run record:")
    print(json.dumps(results[0], indent=2))
    print("\nAggregated result:")
    print(json.dumps(aggregated_result, indent=2))

    print(json.dumps(results[0], indent=2))

if __name__ == "__main__":
    demo()

