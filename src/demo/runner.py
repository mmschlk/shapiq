import json

from aggregator import aggregate_run_records
from custom_types import InteractionIndex, MetricFunction
from experiment_runner import run_experiment
from ground_truth_computer import compute_ground_truth
from metrics import mse_metric, mae_metric
from shapiq.approximator import ProxySHAP, Approximator
from shapiq_games.synthetic import SOUM


def runner() -> None:
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
    results = run_experiment(
        game=game,
        game_name="SOUM", #TODO: determine dynamically
        game_params={
            "n_basis_games": 20,
            "min_interaction_size": 1,
            "max_interaction_size": max_order,
            "random_state": game_seed,
        }, #TODO: determine dynamically
        game_seed=game_seed,
        ground_truth=ground_truth,
        approximator_class=approximator_class,
        index=index,
        max_order=max_order,
        budget=budget,
        approx_seeds=approx_seeds,
        metrics=metrics,
    )
    #aggregation
    aggregated_result = aggregate_run_records(results)


    print("First raw run record:")
    print(json.dumps(results[0], indent=2))
    print("\nAggregated result:")
    print(json.dumps(aggregated_result, indent=2))
    print(json.dumps(results[0], indent=2))

if __name__ == "__main__":
    runner()

