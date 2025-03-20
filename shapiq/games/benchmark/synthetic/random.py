"""This module contains the Random Game which returns a random vector of integers between 0 and 100."""

import numpy as np

from shapiq.games.base import Game


class RandomGame(Game):
    """The RandomGame class returns a random vector of integers between 0 and 100.

    Args:
        n: The number of players.
        random_state: The random state for the random number generator.
    """

    def __init__(self, n: int, random_state: int | None = None):
        self.n = n
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        super().__init__(n, normalize=False)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Returns a random vector of integers between 0 and 100.

        Args:
            coalitions: The coalition as a binary vector of shape (coalition_size, n).

        Returns:
            A random vector of integers between 0 and 100.
        """
        if self.random_state is not None:
            self.rng = np.random.default_rng(self.random_state)
        return self.rng.integers(0, 101, size=coalitions.shape[0])
