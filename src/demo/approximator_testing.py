from shapiq import InteractionValues
from shapiq.game import Game
from shapiq_games.synthetic import DummyGame
from shapiq_games.synthetic import RandomGame
from shapiq_games.synthetic import SOUM
from shapiq.approximator import ProxySHAP, Approximator
from shapiq import ExactComputer #used to compute GT
import numpy as np
from typing import Literal

InteractionIndex = Literal[
    "SV",
    "BV",
    "SII",
    "BII",
    "k-SII",
    "STII",
    "FBII",
    "FSII",
    "kADD-SHAP",
    "CHII",
]

from typing import Callable
from shapiq import InteractionValues

MetricFunction = Callable[[InteractionValues, InteractionValues], float]

def interaction_values_to_dict(
    values: InteractionValues,
) -> dict[tuple[int, ...], float]:
    result = {}
    for interaction, pos in values.interaction_lookup.items():
        result[interaction] = float(values.values[pos])
    return result


def get_common_non_empty_interactions(
    ground_truth: InteractionValues,
    approximation: InteractionValues,
) -> list[tuple[int, ...]]:
    gt_dict = interaction_values_to_dict(ground_truth)
    approx_dict = interaction_values_to_dict(approximation)

    return [
        interaction
        for interaction in gt_dict
        if interaction != () and interaction in approx_dict
    ]


def mse_metric(
    ground_truth: InteractionValues,
    approximation: InteractionValues,
) -> float:
    gt_dict = interaction_values_to_dict(ground_truth)
    approx_dict = interaction_values_to_dict(approximation)
    common = get_common_non_empty_interactions(ground_truth, approximation)
    if not common:
        raise ValueError("No common interactions found.")
    gt_array = np.array([gt_dict[i] for i in common])
    approx_array = np.array([approx_dict[i] for i in common])
    return float(np.mean((gt_array - approx_array) ** 2))


def mae_metric(
    ground_truth: InteractionValues,
    approximation: InteractionValues,
) -> float:
    gt_dict = interaction_values_to_dict(ground_truth)
    approx_dict = interaction_values_to_dict(approximation)
    common = get_common_non_empty_interactions(ground_truth, approximation)
    if not common:
        raise ValueError("No common interactions found.")
    gt_array = np.array([gt_dict[i] for i in common])
    approx_array = np.array([approx_dict[i] for i in common])
    return float(np.mean(np.abs(gt_array - approx_array)))

def single_run(
        game: Game,
        index: InteractionIndex,
        max_order: int,
        approximator_class: type[Approximator],
        budget: int,
        seed: int,
        metrics: dict[str, MetricFunction]
) -> dict[str, float]:
    approximator: Approximator = approximator_class(n=game.n_players,
                                      index=index,
                                      max_order=max_order,
                                      random_state=seed)
    approx_values: InteractionValues = approximator.approximate(budget, game)
    #approx_values.plot_upset()
    exact = ExactComputer(game=game)
    ground_truth = exact.__call__(index=index, order=max_order)
    #ground_truth.plot_upset()
    gt_dict = {}
    for interaction, pos in ground_truth.interaction_lookup.items():
        gt_dict[interaction] = ground_truth.values[pos]

    approx_dict = {}
    for interaction, pos in approx_values.interaction_lookup.items():
        approx_dict[interaction] = approx_values.values[pos]

    common_interactions = []
    for interaction in gt_dict:
        if interaction != () and interaction in approx_dict:
            common_interactions.append(interaction)

    gt_array = np.array([gt_dict[i] for i in common_interactions])
    approx_array = np.array([approx_dict[i] for i in common_interactions])

    metric_results = {}

    for metric_name, metric_func in metrics.items():
        metric_results[metric_name] = metric_func(
            ground_truth,
            approx_values,
        )

    return metric_results

def demo():
    print("This is a test to see how game approximation works")

    # Define the values
    seed = 42
    max_order = 2
    # select index (certain indices like SV expect specific order(1)! )
    # We probably also need to check which approximator supports which index.
    index = "SII"
    game = SOUM(n=10, n_basis_games=20, min_interaction_size=1,
                max_interaction_size=max_order, random_state=seed)
    #We only provide the approximator class to build approximators with different seeds
    approximator_class = ProxySHAP
    budget = 100

    #metrics are just functions for now, this should be refined into a clean easy to work with structure
    #maybe build a metric object
    metrics: dict[str, MetricFunction] = {
        "mse": mse_metric,
        "mae": mae_metric,
    }
    metric_results: dict[str, float] = single_run(game, index, max_order, approximator_class, budget, seed, metrics)

    print("Metric results:")
    for metric_name, metric_value in metric_results.items():
        print(f"{metric_name}: {metric_value}")

demo()