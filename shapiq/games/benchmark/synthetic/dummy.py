"""Provides a simple cooperative game used for testing and benchmarking.

The DummyGame returns the size of a coalition relative to the total number of players, optionally
including an interaction term. It is designed to verify the behavior of algorithms operating on
cooperative games.
"""

from __future__ import annotations

import numpy as np

from shapiq.games.base import Game


class DummyGame(Game):
    """Dummy game for testing purposes.

    When called, the `DummyGame` returns the size of the coalition divided by the number of players
    plus an additional (optional) interaction term.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        interaction: The interaction of the game as a tuple of player indices.
        access_counter: The number of times the game has been called.

    Examples:
        >>> game = DummyGame(4, interaction=(1, 2))
        >>> coalitions = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]]
        >>> coalitions = np.array(coalitions).astype(bool)
        >>> game(coalitions)
        array([0., 0.25 , 1.5, 2])

    """

    def __init__(self, n: int, interaction: set | tuple = ()) -> None:
        """Initializes the DummyGame class.

        Args:
            n: The number of players.
            interaction: The interaction of the game as a tuple of player indices. Defaults to an
                empty tuple.
        """
        self.n = n
        self.N = set(range(self.n))
        self.interaction: tuple = tuple(sorted(interaction))
        self.access_counter = 0
        # init base game class
        super().__init__(n, normalize=False)
        self.access_counter = 0

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the size of the coalition divided by the number of players plus the interaction term.

        Args:
            coalitions: The coalition as a binary vector of shape (coalition_size, n).

        Returns:
            The worth of the coalition.

        """
        worth = np.sum(coalitions, axis=1) / self.n
        if len(self.interaction) > 0:
            interaction = coalitions[:, self.interaction]
            worth += np.prod(interaction, axis=1)
        # update access counter given rows in coalition
        self.access_counter += coalitions.shape[0]
        return worth
