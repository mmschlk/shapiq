from shapiq import ExactComputer, InteractionValues
from shapiq.approximator import Approximator
from shapiq.game import Game
from custom_types import InteractionIndex, MetricFunction

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
    ground_truth: InteractionValues = exact.__call__(index=index, order=max_order)
    #ground_truth.plot_upset()

    metric_results = {}

    for metric_name, metric_func in metrics.items():
        metric_results[metric_name] = metric_func(
            ground_truth,
            approx_values,
        )

    return metric_results