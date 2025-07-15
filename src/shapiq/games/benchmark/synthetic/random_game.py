"""This module contains the Random Game which returns a random vector of integers between 0 and 100."""

from __future__ import annotations

import numpy as np

from shapiq.games.base import Game


class RandomGame(Game):
    """The RandomGame class.

    The RandomGame class is a synthetic benchmark game returning a random vector of integers
    between 0 and 100 for a set of coalitions. The game is used to test the performance of different
    algorithms.

    Attributes:
        n: The number of players.
        random_state: The random state for the random number generator.
        rng: The random number generator.
    """

    def __init__(self, n: int, random_state: int | None = None) -> None:
        """Initializes the RandomGame class.

        Args:
            n: The number of players.
            random_state: The random state for the random number generator. Defaults to ``None``.
        """
        self.n = n
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        super().__init__(n, normalize=False)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Returns a random vector of integers between 0 and 100.

        Args:
            coalitions: The coalition as a binary vector of shape ``(coalition_size, n)``.

        Returns:
            A random vector of integers between 0 and 100.
        """
        if self.random_state is not None:
            self.rng = np.random.default_rng(self.random_state)
        return self.rng.integers(0, 101, size=coalitions.shape[0])
