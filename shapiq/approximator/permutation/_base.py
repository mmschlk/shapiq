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
        random_state: Optional[int] = None,
    ) -> None:
        if index not in AVAILABLE_INDICES_PERMUTATION:
            raise ValueError(
                f"Index {index} is not valid. "
                f"Available indices are {AVAILABLE_INDICES_PERMUTATION}."
            )
        super().__init__(n, max_order, index, top_order, random_state)
        self._iteration_cost: int = -1

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray]
    ) -> InteractionValues:
        """Approximates the interaction values."""
        raise NotImplementedError

    @staticmethod
    def _get_n_iterations(budget: int, batch_size: int, iteration_cost: int) -> tuple[int, int]:
        """Computes the number of iterations and the size of the last batch given the batch size and
        the budget.

        Args:
            budget: The budget for the approximation.
            batch_size: The size of the batch.
            iteration_cost: The cost of a single iteration.

        Returns:
            int, int: The number of iterations and the size of the last batch.
        """
        n_iterations = budget // (iteration_cost * batch_size)
        last_batch_size = batch_size
        remaining_budget = budget - n_iterations * iteration_cost * batch_size
        if remaining_budget > 0 and remaining_budget // iteration_cost > 0:
            last_batch_size = remaining_budget // iteration_cost
            n_iterations += 1
        return n_iterations, last_batch_size

    @property
    def iteration_cost(self) -> int:
        """The cost of a single iteration of the permutation sampling mechanism."""
        return self._iteration_cost
