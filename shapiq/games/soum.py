"""This module contains the SOUM class. The SOUM class is constructed from a linear combination of the UnanimityGame Class.
"""

import numpy as np
from scipy.special import binom

# from shapiq import InteractionValues
# TODO: Convert to InteractionValues object


def transform_dict_to_interaction_values(rslt_dict):
    rslt = np.zeros(len(rslt_dict))
    rslt_pos = {}
    for i, (set, val) in enumerate(rslt_dict.items()):
        rslt[i] = val
        rslt_pos[set] = i
    return rslt, rslt_pos


class UnanimityGame:
    """Unanimity game as basis game in cooperative game theory based on single interaction.
    When called, it returns 1, if the coalition contains the interaction, and 0 otherwise.

    Args:
        interaction: The interaction of the game as a binary vector of shape (n,)

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        interaction: The interaction of the game as a tuple of player indices.
        interaction_binary: Binary representation of the tuple.
    """

    def __init__(self, interaction_binary: np.ndarray):
        self.n = len(interaction_binary)
        self.N = set(range(self.n))
        self.interaction_binary: np.ndarray = interaction_binary
        self.interaction: tuple = tuple(np.where(self.interaction_binary == 1)[0])

    def __call__(self, coalition: np.ndarray) -> np.ndarray[float]:
        """Returns 1, if the coalition contains self.interaction, and zero otherwise.

        Args:
            coalition: The coalition as a binary vector of shape (n,) or (batch_size, n).

        Returns:
            The worth of the coalition.
        """
        if coalition.ndim == 1:
            coalition = coalition.reshape((1, self.n))
        worth = np.prod(coalition >= self.interaction_binary, 1)
        return worth


class SOUM:
    """The SOUM constructs a game based on linear combinations of instances of UnanimityGames.
    When called, it returns 1, if the coalition contains the interaction, and 0 otherwise.

    Args:
        n: The number of players.
        n_basis_games: The number of UnanimityGames.


    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        interaction: The interaction of the game as a tuple of player indices.
        interaction_binary: Binary representation of the tuple.
    """

    def __init__(
        self,
        n,
        n_basis_games: int,
        min_interaction_size: int = None,
        max_interaction_size: int = None,
    ):
        self.n = n
        self.N = set(range(self.n))
        if min_interaction_size == None:
            self.min_interaction_size: int = 0
        else:
            self.min_interaction_size: int = min_interaction_size
        if max_interaction_size == None:
            self.max_interaction_size: int = self.n
        else:
            self.max_interaction_size: int = max_interaction_size

        self.n_basis_games: int = n_basis_games

        self.unanimity_games = {}
        self.linear_coefficients = np.random.random(size=self.n_basis_games)
        # Compute interaction sizes (exclude size 0)
        interaction_sizes = np.random.randint(low=1, high=self.n, size=self.n_basis_games)
        for i, size in enumerate(interaction_sizes):
            interaction = np.random.choice(tuple(self.N), size, replace=False)
            interaction_binary = np.zeros(self.n)
            interaction_binary[interaction] = 1
            self.unanimity_games[i] = UnanimityGame(interaction_binary)

        # Compute the Möbius transform
        self.moebius_transform()

    def __call__(self, coalition: np.ndarray) -> np.ndarray[float]:
        """Returns 1, if the coalition contains self.interaction, and zero otherwise.

        Args:
            coalition: The coalition as a binary vector of shape (n,) or (batch_size, n).

        Returns:
            The worth of the coalition.
        """
        if coalition.ndim == 1:
            coalition = coalition.reshape((1, self.n))

        worth = 0
        for i, game in self.unanimity_games.items():
            worth += self.linear_coefficients[i] * game.__call__(coalition)
        return worth

    def moebius_transform(self):
        """
        Computes the Möbius transform from the UnanimityGames
        """
        self.moebius_coefficients_dict = {}
        for i, game in self.unanimity_games.items():
            if game.interaction in self.moebius_coefficients_dict:
                self.moebius_coefficients_dict[game.interaction] += self.linear_coefficients[i]
            else:
                self.moebius_coefficients_dict[game.interaction] = self.linear_coefficients[i]

        (
            self.moebius_coefficients,
            self.moebius_coefficients_lookup,
        ) = transform_dict_to_interaction_values(self.moebius_coefficients_dict)


if __name__ == "__main__":
    n = 5
    interaction = np.random.randint(2, size=n)
    game = UnanimityGame(interaction)
    coalition_matrix = np.random.randint(2, size=(5, n))
    rslt = game.__call__(coalition_matrix)

    soum = SOUM(n, 20)
    soum_rslt = soum.__call__(coalition_matrix)
