"""This module contains the Random Game which always returns a random vector of integers between
0 and 100."""

import numpy as np

from shapiq.games.base import Game


class RandomGame(Game):
    """The RandomGame class returns a random vector of integers between 0 and 100.

    Args:
        n: The number of players.
    """

    def __init__(self, n: int):
        self.n = n
        # init base game class
        super().__init__(n, normalize=False)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Returns a random vector of integers between 0 and 100.

        Args:
            coalitions: The coalition as a binary vector of shape (coalition_size, n).

        Returns:
            A random vector of integers between 0 and 100.
        """
        return np.random.randint(0, 101, size=coalitions.shape[0])
