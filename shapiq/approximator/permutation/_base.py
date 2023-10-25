"""This module contains the base permutation sampling algorithms to estimate SII/nSII and STI."""
from typing import Optional, Callable, Union

import numpy as np

from approximator._base import Approximator, InteractionValues


AVAILABLE_INDICES_PERMUTATION = {"SII", "nSII", "STI"}


class PermutationSampling(Approximator):
    """Permutation sampling approximator. This class contains the permutation sampling algorithm to
    estimate SII/nSII and STI values.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated. Available indices are 'SII', 'nSII', and
            'STI'.
        top_order: Whether to approximate only the top order interactions (`True`) or all orders up
            to the specified order (`False`).

    Attributes:
        n (int): The number of players.
        N (set): The set of players (starting from 0 to n - 1).
        max_order (int): The interaction order of the approximation.
        index (str): The interaction index to be estimated.
        top_order (bool): Whether to approximate only the top order interactions or all orders up to
            the specified order.
        min_order (int): The minimum order of the approximation. If top_order is True, min_order is
            equal to order. Otherwise, min_order is equal to 1.

    """

    def __init__(
            self,
            n: int,
            max_order: int,
            index: str,
            top_order: bool,
            random_state: Optional[int] = None
    ) -> None:
        if index not in AVAILABLE_INDICES_PERMUTATION:
            raise ValueError(
                f"Index {index} is not valid. "
                f"Available indices are {AVAILABLE_INDICES_PERMUTATION}."
            )
        super().__init__(n, max_order, index, top_order, random_state)

    def approximate(
            self,
            budget: int,
            game: Callable[[Union[set, tuple]], float]
    ) -> InteractionValues:
        """Approximates the interaction values."""
        raise NotImplementedError

    def _get_game_values_storage(self, batch_size: int) -> dict[int, np.ndarray]:
        """Returns a dictionary of arrays to store the game values.

        The array for each order has shape (batch_size, n - order + 2). The first column is for the
        empty set, the last column is for the full set, and the remaining columns are for the
        subsets of the specified order.

        Args:
            batch_size: The batch size.

        Returns:
            dict: The dictionary of arrays.
        """
        return {order: np.zeros((batch_size, self.n - order + 2), dtype=float)
                for order in self._order_iterator}
