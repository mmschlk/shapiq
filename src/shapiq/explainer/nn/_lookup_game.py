"""Defines a utility game class created from a dictionary defining the utility function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
from shapiq import Game


class LookupGame(Game):
    """Defines a Game via a dictionary giving the utility function."""

    def __init__(self, n_players: int, utilities: dict[tuple[int, ...], float]) -> None:
        """Initializes the LookupGame."""
        self.characteristic_function = utilities
        super().__init__(
            n_players=n_players,
            normalization_value=self.characteristic_function[()],
        )

    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Defines the worth of a coalition as a lookup in the characteristic function.

        Args:
            coalitions: A 2D array where each row represents a coalition as a binary
                vector (1 for present, 0 for absent).

        Returns:
            A 1D array containing the value of each coalition based on the
                characteristic function.
        """
        output = [
            self.characteristic_function[tuple(np.where(coalition)[0])] for coalition in coalitions
        ]
        return np.array(output)
