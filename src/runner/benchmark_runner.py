import json

from aggregator import aggregate_run_records
from custom_types import InteractionIndex
from experiment_runner import run_experiment
from ground_truth_computer import compute_ground_truth
from shapiq.approximator import Approximator
from shapiq_games.synthetic import SOUM


def run_benchmark(
    *,
    game_seed: int,
    max_order: int,
    number_of_different_approx_seeds: int,
    budget: int,
    n_players: int,
    n_basis_games: int,
    min_interaction_size: int,
    index: InteractionIndex,
    approximator_class: type[Approximator],
) -> None:

    # Define the values
    approx_seeds = range(number_of_different_approx_seeds)
    #TODO: index approximator validation (e.g. certain indices like SV expect specific order(1)! )
    game = SOUM(n=n_players, n_basis_games=n_basis_games, min_interaction_size=min_interaction_size,
                max_interaction_size=max_order, random_state=game_seed)

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
    )

    #debugging
    # for record in results:
    #     print("failed:", record["run_failed"])
    #     print("error:", record["error_message"])
    #     print()


    #aggregation
    aggregated_result = aggregate_run_records(results)

    print("First raw run record:")
    print(json.dumps(results[0], indent=2))
    print("\nAggregated result:")
    print(json.dumps(aggregated_result, indent=2))
