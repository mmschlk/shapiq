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
    """Approximate interaction values for a game using the given approximator.

    Args:
        game: The game for which interaction values are approximated.
        index: The interaction index to approximate.
        max_order: The maximum interaction order to compute.
        approximator_class: The approximator class used for the approximation.
        budget: The evaluation budget available to the approximator.
        seed: The random seed used for reproducibility.

    Returns:
        The approximated interaction values.
    """
    approximator: Approximator = approximator_class(n=game.n_players,
                                      index=index,
                                      max_order=max_order,
                                      random_state=seed)
    approx_values = approximator.approximate(budget, game)
    #approx_values.plot_upset()
    return approx_values



