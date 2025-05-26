"""This module contains fixtures for the tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def cooking_game():
    """Return the cooking game object."""
    from shapiq.games.base import Game

    class CookingGame(Game):
        def __init__(self):
            self.characteristic_function = {
                (): 10,
                (0,): 4,
                (1,): 3,
                (2,): 2,
                (0, 1): 9,
                (0, 2): 8,
                (1, 2): 7,
                (0, 1, 2): 15,
            }
            super().__init__(
                n_players=3,
                player_names=["Alice", "Bob", "Charlie"],  # Optional list of names
                normalization_value=self.characteristic_function[()],  # 0
                normalize=False,
            )

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            """Defines the worth of a coalition as a lookup in the characteristic function."""
            output = []
            for coalition in coalitions:
                output.append(self.characteristic_function[tuple(np.where(coalition)[0])])
            return np.array(output)

    return CookingGame()


@pytest.fixture
def paper_game():
    """Return a simple game object."""
    from scipy.special import binom

    from shapiq.games.base import Game

    class PaperGame(Game):
        """A simple game with 11 players.

        A game with 11 players, where each coalition must contain at least 2 players and with
        probability 0.1 of two players not cooperating.
        """

        def __init__(self):
            super().__init__(
                n_players=11,
                player_names=None,  # Optional list of names
                normalization_value=0,  # 0
            )

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            """Defines the worth of a coalition as a lookup in the characteristic function."""
            output = [
                sum(coalition) - 0.1 * binom(sum(coalition), 2) if sum(coalition) > 1 else 0
                for coalition in coalitions
            ]
            return np.array(output)

    return PaperGame()
