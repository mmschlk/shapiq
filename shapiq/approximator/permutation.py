"""This module contains the permutation sampling algorithms to estimate SII/nSII and STI."""
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.special import binom

from shapiq.approximator._base import Approximator, InteractionValues
from shapiq.utils import powerset

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

    def __init__(self, n: int, max_order: int, index: str, top_order: bool):
        if index not in AVAILABLE_INDICES_PERMUTATION:
            raise ValueError(
                f"Index {index} is not valid. "
                f"Available indices are {AVAILABLE_INDICES_PERMUTATION}."
            )
        super().__init__(n, max_order, index, top_order)

    def approximate(
            self,
            budget: int,
            game: Callable[[Union[set, tuple]], float]
    ) -> InteractionValues:
        """Approximates the interaction values."""
        raise NotImplementedError


class PermutationSamplingSII(PermutationSampling):

    def __init__(self, n: int, max_order: int, index: str, top_order: bool):
        super().__init__(n, max_order, index, top_order)
        self._iteration_cost: int = self._compute_iteration_cost()

    def _compute_iteration_cost(self) -> int:
        """Computes the cost of performing a single iteration of the permutation sampling given
        the order, the number of players, and the SII index.

        Returns:
            int: The cost of a single iteration.
        """
        iteration_cost: int = 0
        for s in range(self.min_order, self.max_order + 1):
            iteration_cost += (self.n - s + 1) * 2 ** s
        return iteration_cost


class PermutationSamplingSTI(PermutationSampling):

    def __init__(self, n: int, max_order: int, top_order: bool):
        super().__init__(n, max_order, "STI", top_order)
        self._iteration_cost: int = self._compute_iteration_cost()

    def _compute_iteration_cost(self) -> int:
        """Computes the cost of performing a single iteration of the permutation sampling given
        the order, the number of players, and the STI index.

        Returns:
            int: The cost of a single iteration.
        """
        iteration_cost = int(binom(self.n, self.max_order) * 2 ** self.max_order)
        return iteration_cost

    def _compute_lower_order_sti(
            self,
            game: Callable[[Union[set, tuple]], float],
            result: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        """Computes all lower order interactions for the STI index up to order max_order - 1.

        Args:
            game: The game function as a callable that takes a set of players and returns the value.
            result: The result dictionary.

        Returns:
            The result dictionary.
        """
        # run the game on the whole powerset of players up to order max_order - 1
        game_evaluations = {subset: game(subset)
                            for subset in powerset(self.N, max_size=self.max_order - 1, min_size=1)}
        # inspect all parts of the subsets contained in the powerset and attribute their
        # contribution to the corresponding interactions and order
        for subset in powerset(self.N, max_size=self.max_order - 1, min_size=1):
            subset = tuple(subset)
            subset_size = len(subset)
            for subset_part in powerset(subset):
                subset_part_size = len(subset_part)
                update = (-1) ** (subset_size - subset_part_size) * game_evaluations[subset_part]
                result[subset_size][subset] += update
        return result

    def approximate(
            self,
            budget: int,
            game: Callable[[Union[set, tuple]], float]
    ) -> InteractionValues:
        result = self._init_result()
        counts = self._init_result(dtype=int)
        value_empty = game(set())
        value_full = game(self.N)

        lower_order_cost = sum(int(binom(self.n, s)) for s in range(self.min_order, self.max_order))
        if self.max_order > 1 and budget >= lower_order_cost:
            budget -= lower_order_cost
            result = self._compute_lower_order_sti(game, result)

        return self._finalize_result(result)

