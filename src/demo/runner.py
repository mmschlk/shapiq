from shapiq import ExactComputer, InteractionValues
from shapiq.approximator import Approximator
from shapiq.game import Game
from custom_types import InteractionIndex, MetricFunction



def compute_ground_truth(
    game: Game,
    index: InteractionIndex,
    max_order: int,
) -> InteractionValues:
    exact = ExactComputer(game=game)
    return exact(index=index, order=max_order)
    #ground_truth.plot_upset()


def approximate(
        game: Game,
        index: InteractionIndex,
        max_order: int,
        approximator_class: type[Approximator],
        budget: int,
        seed: int
) -> InteractionValues:
    approximator: Approximator = approximator_class(n=game.n_players,
                                      index=index,
                                      max_order=max_order,
                                      random_state=seed)
    approx_values = approximator.approximate(budget, game)
    #approx_values.plot_upset()
    return approx_values



def compute_metrics(
    ground_truth: InteractionValues,
    approximation: InteractionValues,
    metrics: dict[str, MetricFunction],
) -> dict[str, float]:
    metric_results: dict[str, float] = {}

    for metric_name, metric_func in metrics.items():
        metric_results[metric_name] = metric_func(
            ground_truth,
            approximation,
        )

    return metric_results