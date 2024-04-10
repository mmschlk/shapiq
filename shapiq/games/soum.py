"""This module contains the SOUM class. The SOUM class is constructed from a linear combination of the UnanimityGame Class.
"""

import numpy as np
from shapiq.interaction_values import InteractionValues


def _transform_dict_to_interaction_values(rslt_dict):
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
        budget: The budget for the approximation (i.e., the number of game evaluations).
        game: The game function as a callable that takes a set of players and returns the value.
        batch_size: The size of the batch. If None, the batch size is set to `budget`.
            Defaults to None.
        pairing: Whether to use pairwise sampling (`True`) or not (`False`). Defaults to `True`.
            Paired sampling can increase the approximation quality.
        replacement: Whether to sample with replacement (`True`) or without replacement
            (`False`). Defaults to `True`.

    Attributes:
        n: The number of players.
        N: A set of n players
        interaction_binary: The interaction encoded in a binary vector
        interaction: The interaction encoded as a tuple
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
        min_interaction_size: Smallest interaction size, if None then set to zero
        max_interaction_size: Highest interaction size, if None then set to n

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        n_basis_games: The number of Unanimity gams
        unanimity_games: A dictionary containing instances of UnanimityGame
        linear_coefficients: A numpy array with coefficients between -1 and 1 for the unanimity games
        min_interaction_size: The smallest interaction size
        max_interaction_size: The highest interaction size.
        moebius_coefficients: The list of non-zero Möbius coefficients used to compute ground truth values
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
        self.linear_coefficients = np.random.random(size=self.n_basis_games) * 2 - 1
        # Compute interaction sizes (exclude size 0)
        interaction_sizes = np.random.randint(
            low=self.min_interaction_size, high=self.max_interaction_size, size=self.n_basis_games
        )
        for i, size in enumerate(interaction_sizes):
            interaction = np.random.choice(tuple(self.N), size, replace=False)
            interaction_binary = np.zeros(self.n)
            interaction_binary[interaction] = 1
            self.unanimity_games[i] = UnanimityGame(interaction_binary)

        # Compute the Möbius transform
        self.moebius_coefficients = self.moebius_transform()

    def __call__(self, coalition: np.ndarray) -> np.ndarray[float]:
        """Computes the worth of the coalition for the SOUM, i.e. sums up all linear coefficients, if coalition contains the interaction of the corresponding unanimity game.

        Args:
            coalition: The coalition as a binary vector of shape (n,) or (batch_size, n).

        Returns:
            The worth of the coalition.
        """
        if coalition.ndim == 1:
            coalition = coalition.reshape((1, self.n))

        worth = 0
        for i, game in self.unanimity_games.items():
            worth += self.linear_coefficients[i] * game(coalition)
        return worth

    def moebius_transform(self):
        """Computes the (sparse) Möbius transform of the SOUM from the UnanimityGames. Used for ground truth calculations for interaction inidices.

        Args:

        Returns:
            An InteractionValues object containing all non-zero Möbius coefficients of the SOUM.
        """
        moebius_coefficients_dict = {}
        for i, game in self.unanimity_games.items():
            if game.interaction in moebius_coefficients_dict:
                moebius_coefficients_dict[game.interaction] += self.linear_coefficients[i]
            else:
                moebius_coefficients_dict[game.interaction] = self.linear_coefficients[i]

        moebius_coefficients_values = np.zeros(len(moebius_coefficients_dict))
        moebius_coefficients_lookup = {}
        for i, (key, val) in enumerate(moebius_coefficients_dict.items()):
            moebius_coefficients_values[i] = val
            moebius_coefficients_lookup[key] = i

        baseline_value = 0 if tuple() not in moebius_coefficients_dict else None

        moebius_coefficients = InteractionValues(
            values=moebius_coefficients_values,
            index="Moebius",
            max_order=self.n,
            min_order=0,
            n_players=self.n,
            interaction_lookup=moebius_coefficients_lookup,
            estimated=False,
            baseline_value=baseline_value,
        )

        return moebius_coefficients
