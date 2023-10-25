"""This module implements the Permutation Sampling approximator for the SII (and nSII) index.

# TODO add docstring
"""
from typing import Optional, Callable, Union

import numpy as np

from approximator._base import InteractionValues
from approximator.permutation import PermutationSampling
from utils import powerset


class PermutationSamplingSII(PermutationSampling):
    """ Permutation Sampling approximator for the SII (and nSII) index.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        top_order: Whether to approximate only the top order interactions (`True`) or all orders up
            to the specified order (`False`).
        optimize_budget: Whether to optimize (`True`) the budget by reusing the game evaluations
                in the permutation chain or not (`False`). Defaults to `True`.
        random_state: The random state to use for the permutation sampling. Defaults to `None`.
    """

    def __init__(
            self,
            n: int,
            max_order: int,
            top_order: bool,
            optimize_budget: bool = True,
            random_state: Optional[int] = None
    ) -> None:
        super().__init__(n, max_order, 'SII', top_order, random_state)
        self._optimize_budget: bool = optimize_budget
        self._iteration_cost: int = self._compute_iteration_cost()

    def _compute_iteration_cost(self) -> int:
        """Computes the cost of performing a single iteration of the permutation sampling given
        the order, the number of players, and the SII index.

        Returns:
            int: The cost of a single iteration.
        """
        iteration_cost: int = 0
        for s in self._order_iterator:
            iteration_cost += (self.n - s + 1) * 2 ** s
        # TODO add more efficient implementation with self._optimize_budget
        return iteration_cost

    def approximate(
            self,
            budget: int,
            game: Callable[[Union[set, tuple]], float],
            batch_size: Optional[int] = None,
    ) -> InteractionValues:
        """Approximates the interaction values.

        Args:
            budget: The budget for the approximation.
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If None, the batch size is set to 1.

        Returns:
            InteractionValues: The interaction values.
        """
        self._rng = np.random.default_rng(seed=self._random_state)

        batch_size = 1 if batch_size is None else batch_size

        result = self._init_result()
        counts = self._init_result(dtype=int)
        #seen_in_iteration = self._init_result(dtype=bool)  # helper to count subsets per permutation
        value_empty = game(set())
        value_full = game(self.N)

        n_iterations = budget // (self._iteration_cost * batch_size)

        # main permutation sampling loop
        for iteration in range(n_iterations):
            # TODO batch_size can be smaller in the last iteration
            # create permutations: a 2d matrix of shape (batch_size, n) where each row is a
            # permutation of the players
            permutations = np.tile(np.arange(self.n), (batch_size, 1))
            self._rng.permuted(permutations, axis=1, out=permutations)

            # get game values for all subsets of the permutations
            game_value_storage = self._get_game_values_storage(batch_size)
            for order in self._order_iterator:
                game_value_storage[order][:, 0] = value_empty
                game_value_storage[order][:, -1] = value_full
                for k in range(self.n - order + 1):
                    coalitions = permutations[:, :k + order]
                    coalitions_values = game(coalitions)  # TODO make game functions accept arrays
                    game_value_storage[order][:, k] = coalitions_values

            # compute interaction scores and update result
            for order in self._order_iterator:
                for k in range(self.n - order + 1):
                    coalitions = permutations[:, :k + order]
                    game_values = game_value_storage[order][:, k]
                    for coalition, game_value in zip(coalitions, game_values):
                        seen_in_iteration = self._get_empty_array(n=self.n, order=order, dtype=bool)
                        coalition = tuple(sorted(coalition))
                        for coalition_parts in powerset(coalition, min_size=1, max_size=order):
                            update = game_value * (-1) ** (order - len(coalition_parts))
                            result[order][coalition_parts] += update
                            seen_in_iteration[coalition_parts] = True
                        counts[order] += seen_in_iteration

        # compute mean of interactions
        for s in self._order_iterator:
            result[s] = np.divide(result[s], counts[s], out=result[s], where=counts[s] != 0)

        return self._finalize_result(result)


if __name__ == "__main__":
    from games.dummy import DummyGame

    n_run = 7

    game_run = DummyGame(n_run, (1, 2))

    approximator = PermutationSamplingSII(n_run, 2, False)

    result_run: InteractionValues = approximator.approximate(10000, game_run, batch_size=100)

    # pretty print result
    for coalition_run in powerset(set(range(n_run)), max_size=2, min_size=1):
        print(coalition_run, round(result_run.values[len(coalition_run)][tuple(sorted(coalition_run))], 3))