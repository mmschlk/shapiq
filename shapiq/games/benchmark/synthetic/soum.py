"""This module contains the SOUM class. The SOUM class is constructed from a linear combination of
the UnanimityGame Class."""

from typing import Optional

import numpy as np

from shapiq.games import Game
from shapiq.interaction_values import InteractionValues


class UnanimityGame(Game):
    """Unanimity game as basis game in cooperative game theory based on single interaction.
    When called, it returns 1, if the coalition contains the interaction, and 0 otherwise.

    Args:
        interaction_binary: The interaction encoded in a binary vector of shape (n,).

    Attributes:
        n_players: The number of players.
        interaction_binary: The interaction encoded in a binary vector
        interaction: The interaction encoded as a tuple

    Examples:
        >>> game = UnanimityGame(np.array([0, 1, 0, 1]))
        >>> game.n_players
        4
        >>> coalitions = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]]
        >>> coalitions = np.array(coalitions).astype(bool)
        >>> game(coalitions)
        array([0., 0., 1., 1.])
    """

    def __init__(self, interaction_binary: np.ndarray):
        n = len(interaction_binary)
        self.interaction_binary: np.ndarray = interaction_binary
        self.interaction: tuple[int, ...] = tuple(np.where(self.interaction_binary == 1)[0])
        super().__init__(n_players=n, normalize=False)  # call super class which handles calls

    def value_function(self, coalitions: np.ndarray) -> np.ndarray[float]:
        """Returns 1, if the coalition contains the UnanimityGame's interaction, and zero otherwise.

        Args:
            coalitions: The coalition as a binary vector of shape (coalition_size, n_players).

        Returns:
            The worth of the coalition.
        """
        worth = np.prod(coalitions >= self.interaction_binary, 1)
        return worth


class SOUM(Game):
    """The SOUM constructs a game based on linear combinations of instances of UnanimityGames.
    When called, it returns 1, if the coalition contains the interaction, and 0 otherwise.

    Args:
        n: The number of players.
        n_basis_games: The number of UnanimityGames.
        min_interaction_size: Smallest interaction size, if None then set to zero
        max_interaction_size: Highest interaction size, if None then set to n

    Attributes:
        n_players: The number of players.
        n_basis_games: The number of Unanimity gams
        unanimity_games: A dictionary containing instances of UnanimityGame
        linear_coefficients: A numpy array with coefficients between -1 and 1 for the unanimity games
        min_interaction_size: The smallest interaction size
        max_interaction_size: The highest interaction size.
        moebius_coefficients: The list of non-zero Möbius coefficients used to compute ground truth
            values

    Examples:
        >>> game = SOUM(4, 3)
        >>> game.n_players
        4
        >>> game.n_basis_games
        3
        coalitions = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]]
        >>> coalitions = np.array(coalitions).astype(bool)
        >>> game(coalitions)
        array([0., 0.25, 1.5, 2.])  # depending on the random linear coefficients this can vary
        >>> game.moebius_coefficients
        InteractionValues(values=array([0.25, 0.25, 0.25]), index='Moebius', max_order=4, min_order=0, ...)
    """

    def __init__(
        self,
        n: int,
        n_basis_games: int,
        min_interaction_size: Optional[int] = None,
        max_interaction_size: Optional[int] = None,
    ):
        # init base game
        super().__init__(n_players=n, normalize=False)

        # set min_interaction_size and max_interaction_size to 0 and n if not specified
        self.min_interaction_size = min_interaction_size if min_interaction_size is not None else 0
        self.max_interaction_size = max_interaction_size if max_interaction_size is not None else n

        # setup basis games
        self.n_basis_games: int = n_basis_games
        self.unanimity_games = {}
        self.linear_coefficients = np.random.random(size=self.n_basis_games) * 2 - 1
        # Compute interaction sizes (exclude size 0)
        interaction_sizes = np.random.randint(
            low=self.min_interaction_size, high=self.max_interaction_size, size=self.n_basis_games
        )
        for i, size in enumerate(interaction_sizes):
            interaction = np.random.choice(tuple(range(self.n_players)), size, replace=False)
            interaction_binary = np.zeros(self.n_players, dtype=int)
            interaction_binary[interaction] = 1
            self.unanimity_games[i] = UnanimityGame(interaction_binary)

        # Compute the Möbius transform
        self.moebius_coefficients = self.moebius_transform()

    def value_function(self, coalitions: np.ndarray[bool]) -> np.ndarray[float]:
        """Computes the worth of the coalition for the SOUM, i.e. sums up all linear coefficients,
        if coalition contains the interaction of the corresponding unanimity game.

        Args:
            coalitions: The coalition as a binary vector of shape (coalition_size, n).

        Returns:
            The worth of the coalition.
        """
        worth = np.zeros(coalitions.shape[0])
        for i, game in self.unanimity_games.items():
            worth += self.linear_coefficients[i] * game(coalitions)
        return worth

    def moebius_transform(self):
        """Computes the (sparse) Möbius transform of the SOUM from the UnanimityGames. Used for
        ground truth calculations for interaction indices.

        Args:

        Returns:
            An InteractionValues object containing all non-zero Möbius coefficients of the SOUM.
        """
        # fill the moebius coefficients dict from the game
        moebius_coefficients_dict = {}
        for i, game in self.unanimity_games.items():
            try:
                moebius_coefficients_dict[game.interaction] += self.linear_coefficients[i]
            except KeyError:
                moebius_coefficients_dict[game.interaction] = self.linear_coefficients[i]

        # generate the lookup for the moebius values
        moebius_coefficients_values = np.zeros(len(moebius_coefficients_dict))
        moebius_coefficients_lookup = {}
        for i, (key, val) in enumerate(moebius_coefficients_dict.items()):
            moebius_coefficients_values[i] = val
            moebius_coefficients_lookup[key] = i

        # handle baseline value and set to 0 if no empty set is present
        baseline_value = 0 if tuple() not in moebius_coefficients_dict else None

        moebius_coefficients = InteractionValues(
            values=moebius_coefficients_values,
            index="Moebius",
            max_order=self.n_players,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=moebius_coefficients_lookup,
            estimated=False,
            baseline_value=baseline_value,
        )

        return moebius_coefficients