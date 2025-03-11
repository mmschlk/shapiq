"""This module contains fixtures for the tests."""

import numpy as np
import pytest


@pytest.fixture
def cooking_game():
    import shapiq

    """Return a simple game object."""

    class CookingGame(shapiq.Game):
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
