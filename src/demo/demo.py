from shapiq_games.synthetic import SOUM
from shapiq.approximator import ProxySHAP, Approximator
from custom_types import InteractionIndex, MetricFunction
from metrics import mse_metric, mae_metric
from runner import approximate, compute_ground_truth, compute_metrics
from importlib.metadata import version
import time
import numpy as np

def demo() -> None:
    print("This is a test to see how game approximation works")

    # Define the values
    SHAPIQ_VERSION = version("shapiq")
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
        approx_values = approximate(
            game=game,
            approximator_class=approximator_class,
            index=index,
            max_order=max_order,
            budget=budget,
            seed=approx_seed
        )

        #calculate metrics for each run
        metric_results: dict[str, float] = compute_metrics(
            ground_truth=ground_truth,
            approximation=approx_values,
            metrics=metrics
        )

        results.append({
            "game_seed": game_seed,
            "approx_seed": approx_seed,
            "budget": budget,
            "mse": metric_results["mse"],
            "mae": metric_results["mae"],
        })

    #aggregation
    mse_values = np.array([result["mse"] for result in results])
    mae_values = np.array([result["mae"] for result in results])
    aggregated_result ={
        "mse_mean": float(np.mean(mse_values)),
        "mae_mean": float(np.mean(mae_values)),
    }


    print("Metric results:")
    for metric_name, metric_value in aggregated_result.items():
        print(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    demo()

