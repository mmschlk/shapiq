from custom_types import InteractionIndex
from shapiq import InteractionValues
from shapiq.approximator import Approximator
from shapiq.game import Game


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



