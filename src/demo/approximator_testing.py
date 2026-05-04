from shapiq import InteractionValues
from shapiq.game import Game
from shapiq_games.synthetic import DummyGame
from shapiq_games.synthetic import RandomGame
from shapiq_games.synthetic import SOUM
from shapiq.approximator import ProxySHAP
from shapiq import ExactComputer #used to compute GT
import numpy as np

def single_run(game, index, max_order, approximator, budget, seed):
    approx_values = approximator.approximate(budget, game)
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

    mean_squared_error = np.mean((gt_array - approx_array) ** 2)
    print("MSE between approximation and ground truth is: " + str(mean_squared_error))



def demo():
    print("This is a test to see how game approximation works")

    # Define the values
    seed = 42
    max_order = 2
    # select index (certain indices like SV expect specific order(1)! )
    index = "SII"
    game = SOUM(n=10, n_basis_games=20, min_interaction_size=1,
                max_interaction_size=max_order, random_state=seed)
    approximator = ProxySHAP(n=game.n_players, index=index, max_order=max_order)
    budget = 100
    single_run(game, index, max_order, approximator, budget, seed)

demo()