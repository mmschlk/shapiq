"""This example demonstrates the use of a custom game and the exact computer class to compute exact Shapley values and interactions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from shapiq.exact import ExactComputer
from shapiq.games.benchmark.uncertainty.benchmark import AdultCensus

if __name__ == "__main__":
    benchmark = AdultCensus(
        model_name="random_forest", verbose=True, imputer="marginal", random_state=5
    )
    benchmark.precompute()
    print(benchmark.value_storage)
    exact = ExactComputer(n_players=benchmark.n_players, game_fun=benchmark)
    shapley_values = exact(index="SV", order=1)
    print(shapley_values)

    benchmark = AdultCensus(
        model_name="random_forest", verbose=True, imputer="conditional", random_state=5
    )
    benchmark.precompute()
    print(benchmark.value_storage)
    exact = ExactComputer(n_players=benchmark.n_players, game_fun=benchmark)
    shapley_values = exact(index="SV", order=1)
    print(shapley_values)

    # # Initialize ExactComputer - this computes all values exhaustively
    # exact_computer = ExactComputer(n_players=n, game_fun=game)
    # # Compute Shapley values
    # shapley_values = exact_computer(index="SV", order=1)
    # # Compute (pairwise) Shapley Interactions (k-SII)
    # # According to https://www.nature.com/articles/s42256-019-0138-9.
    # # and https://proceedings.mlr.press/v206/bordt23a.html
    # shapley_interactions = exact_computer(index="k-SII", order=2)
    #
    # # The grand coalition value, e.g. the prediction of the model in case of local XAI
    # grand_coalition = game(np.ones(n))
    # # The baseline value, e.g. the random baseline prediction in case of local XAI
    # baseline_value = game(np.zeros(n))
    # print("Grand coalition: ", grand_coalition, "\nBaseline value: ", baseline_value)
    #
    # # The sum of the shapley values or interactions is equal the difference between grand coalition and baseline
    # # The Interaction Values object contains all non-zero Shapley values / interactions and the baseline value
    # print("Sum Shapley Interactions: ", np.sum(shapley_interactions.values))
    # print("Sum SV: ", np.sum(shapley_values.values))
    #
    # print(
    #     "Interaction: ",
    #     game.interaction_tuple,
    #     " with value ",
    #     game.interaction_value,
    #     " and baseline: ",
    #     game.baseline_value,
    # )
    # # Print Shapley values
    # print("Shapley values: ", shapley_values)
    # # Print pairwise Shapley interactions
    # print("Shapley interactions: ", shapley_interactions)
