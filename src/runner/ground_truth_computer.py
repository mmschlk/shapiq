from shapiq import ExactComputer, InteractionValues
from shapiq.game import Game
from custom_types import InteractionIndex

def compute_ground_truth(
    game: Game,
    index: InteractionIndex,
    max_order: int,
) -> InteractionValues:
    """
    Compute exact interaction values for a game.

    Args:
        game: The game for which exact interaction values are computed.
        index: The interaction index to compute.
        max_order: The maximum interaction order to compute.

    Returns:
        The exact interaction values.
    """
    exact = ExactComputer(game=game)
    return exact(index=index, order=max_order)
    #ground_truth.plot_upset()