from typing import Callable, Union

import numpy as np
from scipy.special import binom

from approximator._base import InteractionValues
from approximator.permutation import PermutationSampling
from utils import powerset


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

        # compute all lower order interactions if budget allows it
        lower_order_cost = sum(int(binom(self.n, s)) for s in range(self.min_order, self.max_order))
        if self.max_order > 1 and budget >= lower_order_cost:
            budget -= lower_order_cost
            result = self._compute_lower_order_sti(game, result)

        # main permutation sampling loop
        n_permutations = 0
        while budget >= self._iteration_cost:
            budget -= self._iteration_cost
            values = np.zeros(self.n + 1)  # create array for the current permutation
            values[0], values[-1] = value_empty, value_full  # init values on the edges
            permutation = np.random.permutation(self.n)  # create random permutation
            # TODO finish this

        return self._finalize_result(result)
