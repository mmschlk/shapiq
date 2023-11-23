from typing import Callable
import warnings

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
        iteration_cost = int(binom(self.n, self.max_order) * 2**self.max_order)
        return iteration_cost

    def _compute_lower_order_sti(
        self, game: Callable[[np.ndarray], np.ndarray], result: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        """Computes all lower order interactions for the STI index up to order max_order - 1.

        Args:
            game: The game function as a callable that takes a set of players and returns the value.
            result: The result dictionary.

        Returns:
            The result dictionary.
        """
        # get all game values on the whole powerset of players up to order max_order - 1
        lower_order_sizes = list(range(0, self.max_order))
        subsets: np.ndarray[bool] = self._get_explicit_subsets(self.n, lower_order_sizes)
        game_values = game(subsets)
        game_values_lookup = {
            tuple(np.where(subsets[index])[0]): float(game_values[index])
            for index in range(subsets.shape[0])
        }

        # compute the discrete derivatives of all subsets
        for subset in powerset(self.N, min_size=1, max_size=self.max_order - 1):  # S
            subset_size = len(subset)  # |S|
            for subset_part in powerset(subset):  # L
                subset_part_size = len(subset_part)  # |L|
                game_value = game_values_lookup[subset_part]  # \nu(L)
                update = (-1) ** (subset_size - subset_part_size) * game_value
                result[subset_size][subset] += update
        return result

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], batch_size: int = 1
    ) -> InteractionValues:
        """Approximates the interaction values.

        Args:
            budget: The budget for the approximation.
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If None, the batch size is set to 1. Defaults to 1.

        Returns:
            InteractionValues: The estimated interaction values.
        """
        batch_size = 1 if batch_size is None else batch_size
        used_budget = 0

        result = self._init_result()
        counts = self._init_result(dtype=int)

        # compute all lower order interactions if budget allows it
        lower_order_cost = sum(int(binom(self.n, s)) for s in range(self.min_order, self.max_order))
        if self.max_order > 1 and budget >= lower_order_cost:
            budget -= lower_order_cost
            used_budget += lower_order_cost
            result = self._compute_lower_order_sti(game, result)
        else:
            warnings.warn(
                message=f"The budget {budget} is too small to compute the lower order interactions "
                f"of the STI index, which requires {lower_order_cost} evaluations. Consider "
                f"increasing the budget.",
                category=UserWarning,
            )
            return self._finalize_result(result, budget=used_budget)

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._get_n_iterations(
            budget, batch_size, self._iteration_cost
        )

        # warn the user if the budget is too small
        if n_iterations == 0:
            warnings.warn(
                message=f"The budget {budget} is too small to perform a single iteration, which "
                f"requires {self._iteration_cost + lower_order_cost} evaluations. Consider "
                f"increasing the budget.",
                category=UserWarning,
            )
            return self._finalize_result(result, budget=used_budget)

        # main permutation sampling loop
        for iteration in range(1, n_iterations + 1):
            batch_size = batch_size if iteration != n_iterations else last_batch_size

            # create the permutations: a 2d matrix of shape (batch_size, n) where each row is a
            # permutation of the players
            permutations = np.tile(np.arange(self.n), (batch_size, 1))
            self._rng.permuted(permutations, axis=1, out=permutations)
            n_permutations = permutations.shape[0]
            n_subsets = n_permutations * self._iteration_cost

            # get all subsets to evaluate per iteration
            subsets = np.zeros(shape=(n_subsets, self.n), dtype=bool)
            subset_index = 0
            for permutation_id in range(n_permutations):
                for interaction in powerset(self.N, self.max_order, self.max_order):
                    idx = 0
                    for i in permutations[permutation_id]:
                        if i in interaction:
                            break
                        else:
                            idx += 1
                    subset = tuple(permutations[permutation_id][:idx])
                    for L in powerset(interaction):
                        subsets[subset_index, tuple(subset + L)] = True
                        subset_index += 1

            # evaluate all subsets on the game
            game_values: np.ndarray[float] = game(subsets)

            # update the interaction scores by iterating over the permutations again
            subset_index = 0
            for permutation_id in range(n_permutations):
                for interaction in powerset(self.N, self.max_order, self.max_order):
                    counts[self.max_order][interaction] += 1
                    for L in powerset(interaction):
                        game_value = game_values[subset_index]
                        update = game_value * (-1) ** (self.max_order - len(L))
                        result[self.max_order][interaction] += update
                        subset_index += 1

            used_budget += self._iteration_cost * batch_size

        # compute mean of interactions
        for s in self._order_iterator:
            result[s] = np.divide(result[s], counts[s], out=result[s], where=counts[s] != 0)

        return self._finalize_result(result, budget=used_budget)