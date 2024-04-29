"""This example demonstrates the use of a custom game and the exact computer class to compute exact Shapley values and interactions."""

from shapiq.exact import ExactComputer
import numpy as np
from shapiq.games import Game


class MyGame(Game):
    """A simple custom game. The game initializes a random pairwise interaction with a random interaction and baseline
    value. The worth of a coalition is defined as the sum of the baseline value and the average size of the coalition
    as well as the interaction term, if the coalition contains the interaction.
    """

    def __init__(self, n_players):
        super().__init__(n_players=n_players, normalize=False, verbose=True)
        self.interaction_tuple = tuple(
            np.random.choice(self.n_players, size=min(self.n_players, 2), replace=False)
        )
        self.interaction_value = np.random.random()
        self.baseline_value = np.random.random()

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """This function is mandatory and defines the logic of the game by returning the worth of all coalitions.

        Args:
            coalitions: A binary matrix of shapley (n_coalitions, n_players) indicating the presence of the players.

        Returns:
            The worth of all coalitions.
        """
        # Insert your custom logic here
        worth = np.sum(coalitions, axis=1) / self.n_players
        worth[
            np.sum(coalitions[:, self.interaction_tuple], axis=1) == len(self.interaction_tuple)
        ] += self.interaction_value
        worth += self.baseline_value
        return worth


if __name__ == "__main__":
    n = 10
    # Initialize custom game
    game = MyGame(n)
    # Initialize ExactComputer - this computes all values exhaustively
    exact_computer = ExactComputer(n_players=n, game_fun=game)
    # Compute Shapley values
    shapley_values = exact_computer(index="SV", order=1)
    # Compute (pairwise) Shapley Interactions (k-SII)
    # According to https://www.nature.com/articles/s42256-019-0138-9.
    # and https://proceedings.mlr.press/v206/bordt23a.html
    shapley_interactions = exact_computer(index="k-SII", order=2)

    # The grand coalition value, e.g. the prediction of the model in case of local XAI
    grand_coalition = game(np.ones(n))
    # The baseline value, e.g. the random baseline prediction in case of local XAI
    baseline_value = game(np.zeros(n))
    print("Grand coalition: ", grand_coalition, "\nBaseline value: ", baseline_value)

    # The sum of the shapley values or interactions is equal the difference between grand coalition and baseline
    # The Interaction Values object contains all non-zero Shapley values / interactions and the baseline value
    print("Sum Shapley Interactions: ", np.sum(shapley_interactions.values))
    print("Sum SV: ", np.sum(shapley_values.values))

    print(
        "Interaction: ",
        game.interaction_tuple,
        " with value ",
        game.interaction_value,
        " and baseline: ",
        game.baseline_value,
    )
    # Print Shapley values
    print("Shapley values: ", shapley_values)
    # Print pairwise Shapley interactions
    print("Shapley interactions: ", shapley_interactions)
