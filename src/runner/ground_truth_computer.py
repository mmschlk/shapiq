from shapiq import ExactComputer, InteractionValues
from shapiq.game import Game
from custom_types import InteractionIndex

def compute_ground_truth(
    game: Game,
    index: InteractionIndex,
    max_order: int,
) -> InteractionValues:
    exact = ExactComputer(game=game)
    return exact(index=index, order=max_order)
    #ground_truth.plot_upset()