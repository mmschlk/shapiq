"""This module contains the DummyGame class. The DummyGame class is mainly used for testing
purposes. It returns the size of the coalition divided by the number of players plus an additional
interaction term."""

from typing import Union

import numpy as np


class DummyGame:
    """Dummy game for testing purposes. When called, it returns the size of the coalition divided by
    the number of players plus an additional interaction term.

    Args:
        n: The number of players.
        interaction: The interaction of the game as a tuple of player indices. Defaults to an empty
            tuple.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        interaction: The interaction of the game as a tuple of player indices.
        access_counter: The number of times the game has been called.
    """

    def __init__(self, n: int, interaction: Union[set, tuple] = ()):
        self.n = n
        self.N = set(range(self.n))
        self.interaction: tuple = tuple(sorted(interaction))
        self.access_counter = 0

    def __call__(self, coalition: np.ndarray) -> np.ndarray[float]:
        """Returns the size of the coalition divided by the number of players plus the interaction
        term.

        Args:
            coalition: The coalition as a binary vector of shape (n,) or (batch_size, n).

        Returns:
            The worth of the coalition.
        """
        if coalition.ndim == 1:
            coalition = coalition.reshape((1, self.n))
        worth = np.sum(coalition, axis=1) / self.n
        if len(self.interaction) > 0:
            interaction = coalition[:, self.interaction]
            worth += np.prod(interaction, axis=1)
        # update access counter given rows in coalition
        self.access_counter += coalition.shape[0]
        return worth

    def __repr__(self):
        return f"DummyGame(n={self.n}, interaction={self.interaction})"

    def __str__(self):
        return f"DummyGame(n={self.n}, interaction={self.interaction})"
