"""This module contains the SOUM class. The SOUM class is constructed from a linear combination of
the UnanimityGame Class."""

from typing import Optional

import numpy as np

from shapiq.games.base import Game
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
        random_state: Optional[int] = None,
        normalize: bool = False,
        verbose: bool = False,
    ):
        from ....moebius_converter import MoebiusConverter

        self._rng = np.random.default_rng(random_state)

        # set min_interaction_size and max_interaction_size to 0 and n if not specified
        self.min_interaction_size = min_interaction_size if min_interaction_size is not None else 0
        self.max_interaction_size = max_interaction_size if max_interaction_size is not None else n

        # setup basis games
        self.n_basis_games: int = n_basis_games
        self.unanimity_games = {}
        self.linear_coefficients = self._rng.random(size=self.n_basis_games) * 2 - 1
        # Compute interaction sizes (exclude size 0)
        interaction_sizes = self._rng.integers(
            low=self.min_interaction_size,
            high=self.max_interaction_size,
            size=self.n_basis_games,
            endpoint=True,
        )
        for i, size in enumerate(interaction_sizes):
            interaction = self._rng.choice(tuple(range(n)), size, replace=False)
            interaction_binary = np.zeros(n, dtype=int)
            interaction_binary[interaction] = 1
            self.unanimity_games[i] = UnanimityGame(interaction_binary)

        # will store the Möbius transform
        self._moebius_coefficients: Optional[InteractionValues] = None
        self.converter: Optional[MoebiusConverter] = None

        # init base game
        empty_value = float(self.value_function(np.zeros((1, n)))[0])
        super().__init__(
            n_players=n, normalize=normalize, verbose=verbose, normalization_value=empty_value
        )

    @property
    def moebius_coefficients(self) -> InteractionValues:
        if self._moebius_coefficients is None:
            self._moebius_coefficients = self.moebius_transform()
        return self._moebius_coefficients

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
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

    def exact_values(self, index: str, order: int) -> InteractionValues:
        """Computes the exact values for the given index and order.

        Args:
            index: The index to compute the values for.
            order: The order to compute the values for.

        Returns:
            The exact values for the given index and order.
        """
        from ....moebius_converter import MoebiusConverter

        if self.converter is None:
            self.converter = MoebiusConverter(self.moebius_coefficients)
        values = self.converter(index, order)
        return values

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
        try:
            baseline_value = moebius_coefficients_dict[tuple()]
        except KeyError:
            baseline_value = 0.0

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
