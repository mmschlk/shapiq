from shapiq_games.synthetic import SOUM
from shapiq.approximator import ProxySHAP, Approximator
from custom_types import InteractionIndex, MetricFunction
from metrics import mse_metric, mae_metric
from runner import single_run

def demo() -> None:
    print("This is a test to see how game approximation works")

    # Define the values
    seed = 42
    max_order = 2
    # select index (certain indices like SV expect specific order(1)! )
    # We probably also need to check which approximator supports which index.
    index : InteractionIndex = "SII"
    game = SOUM(n=10, n_basis_games=20, min_interaction_size=1,
                max_interaction_size=max_order, random_state=seed)
    #We only provide the approximator class to build approximators with different seeds
    approximator_class: type[Approximator] = ProxySHAP
    budget = 100

    #metrics are just functions for now, this should be refined into a clean easy to work with structure
    #maybe build a metric object
    metrics: dict[str, MetricFunction] = {
        "mse": mse_metric,
        "mae": mae_metric,
    }
    metric_results: dict[str, float] \
        = single_run(game, index, max_order, approximator_class, budget, seed, metrics)

    print("Metric results:")
    for metric_name, metric_value in metric_results.items():
        print(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    demo()