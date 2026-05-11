import json

from aggregator import aggregate_run_records
from custom_types import InteractionIndex
from experiment_runner import run_experiment
from ground_truth_computer import compute_ground_truth
from shapiq.approximator import Approximator

def run_benchmark(
    *,
    game,
    game_name: str,
    game_params: dict,
    game_seed: int,
    max_order: int,
    number_of_different_approx_seeds: int,
    budget: int,
    index: InteractionIndex,
    approximator_class: type[Approximator],
) -> dict:

    # Define the values
    #TODO: index approximator validation (e.g. certain indices like SV expect specific order(1)! )
    approx_seeds = range(number_of_different_approx_seeds)

    #Compute ground truth
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

    #debugging
    # for record in results:
    #     print("failed:", record["run_failed"])
    #     print("error:", record["error_message"])
    #     print()


    #aggregation
    aggregated_result = aggregate_run_records(results)

    #print-out
    # print("number of raw results:", len(results))
    # print("First raw run record:")
    # print(json.dumps(results[0], indent=2))
    # print("\nAggregated result:")
    # print(json.dumps(aggregated_result, indent=2))

    return {
        "raw_results": results,
        "aggregated_result": aggregated_result,
    }