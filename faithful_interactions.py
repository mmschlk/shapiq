import copy

import numpy as np
from scipy.special import binom

from shapiq import ExactComputer, Game, powerset

# plot the results
from shapiq.games.benchmark import SOUM


class approx_game(Game):
    def __init__(self, n_players, values):
        # init the base game
        super().__init__(
            n_players,
            normalize=False,
            normalization_value=0,
            verbose=False,
        )
        self._grand_coalition_set = set(range(self.n_players))
        self.game_values, self.coalition_lookup_test = self.approximate_game_values(values)

    def approximate_game_values(self, values):
        game_values = np.zeros(2**self.n_players)
        game_lookup = {}
        for i, T in enumerate(powerset(self._grand_coalition_set)):
            game_values[i] = 0
            game_lookup[tuple(T)] = i
            for S in powerset(self._grand_coalition_set, min_size=2, max_size=2):
                if len(set(S).intersection(set(T))) == 1:
                    game_values[i] += values[S]  # *(-0.5)
        return game_values, game_lookup

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        values = np.zeros(np.shape(coalitions)[0])
        for row_index, row in enumerate(coalitions):
            values[row_index] = self.game_values[
                self.coalition_lookup_test[tuple(np.where(row)[0])]
            ]
        return values


def check_sum(values, i):
    output = 0
    for j in range(n_players):
        if i != j:
            output += values[
                tuple(
                    sorted(
                        (
                            i,
                            j,
                        )
                    )
                )
            ]
    return output


def shapley(values, i, k):
    output = 0
    for coal in powerset(N, min_size=k, max_size=k):
        if i not in coal:
            coal_array = np.zeros(n_players, dtype=bool)
            coal_array[list(coal)] = True
            coal_i = copy.copy(coal_array)
            coal_i[i] = True
            output += (
                1
                / n_players
                * binom(n_players - 1, len(coal))
                * (values(coal_i) - values(coal_array))
            )
    return output


if __name__ == "__main__":
    # read these values from the configuration file / or the printed benchmark configurations
    # game_identifier = "SentimentAnalysisLocalXAI"  # explains the sentiment of a sentence
    # game_identifier = "SOUM"
    # config_id = 1
    # n_player_id = 0
    # n_games = 3

    soum_game = SOUM(n=11, n_basis_games=10)

    n_players = soum_game.n_players
    N = set(range(n_players))

    exact_computer = ExactComputer(n_players=n_players, game_fun=soum_game)
    fsii = exact_computer.shapley_interaction("FSII", order=2)

    approx_game_instance = approx_game(n_players, fsii)
    approx_exact = ExactComputer(n_players, approx_game_instance)
    remainder_sv = approx_exact("SV", order=1)
    print(remainder_sv.values)

    for k in range(n_players):
        print(shapley(approx_game_instance, 0, k))
