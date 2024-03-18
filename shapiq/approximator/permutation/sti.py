"""This module contains the permutation sampling algorithms to estimate STI scores."""

import warnings
from typing import Callable, Optional

import numpy as np
from approximator._base import Approximator
from interaction_values import InteractionValues
from scipy.special import binom
from utils import get_explicit_subsets, powerset


class PermutationSamplingSTI(Approximator):
    """Permutation Sampling approximator for the Shapley Taylor Index (STI).

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        random_state: The random state to use for the permutation sampling. Defaults to `None`.

    Attributes:
        n: The number of players.
        max_order: The interaction order of the approximation.
        min_order: The minimum order to approximate.
        iteration_cost: The cost of a single iteration of the permutation sampling.

    Example:
        >>> from games import DummyGame
        >>> from approximator import PermutationSamplingSTI
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = PermutationSamplingSTI(n=5, max_order=2)
        >>> approximator.approximate(budget=200, game=game)
        InteractionValues(
            index=STI, order=2, estimated=True, estimation_budget=165,
            values={
                (0,): 0.2,
                (1,): 0.2,
                (2,): 0.2,
                (3,): 0.2,
                (4,): 0.2,
                (0, 1): 0,
                (0, 2): 0,
                (0, 3): 0,
                (0, 4): 0,
                (1, 2): 1.0,
                (1, 3): 0,
                (1, 4): 0,
                (2, 3): 0,
                (2, 4): 0,
                (3, 4): 0
            }
        )
    """

    def __init__(self, n: int, max_order: int, random_state: Optional[int] = None) -> None:
        super().__init__(n, max_order, index="STI", top_order=False, random_state=random_state)
        self.iteration_cost: int = self._compute_iteration_cost()

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

        result: np.ndarray[float] = self._init_result()
        counts: np.ndarray[int] = self._init_result(dtype=int)

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
            return self._finalize_result(result, budget=used_budget, estimated=True)

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget, batch_size, self.iteration_cost
        )

        # warn the user if the budget is too small
        if n_iterations == 0:
            warnings.warn(
                message=f"The budget {budget} is too small to perform a single iteration, which "
                f"requires {self.iteration_cost + lower_order_cost} evaluations. Consider "
                f"increasing the budget.",
                category=UserWarning,
            )
            return self._finalize_result(result, budget=used_budget, estimated=True)

        # main permutation sampling loop
        for iteration in range(1, n_iterations + 1):
            batch_size = batch_size if iteration != n_iterations else last_batch_size

            # create the permutations: a 2d matrix of shape (batch_size, n) where each row is a
            # permutation of the players
            permutations = np.tile(np.arange(self.n), (batch_size, 1))
            self._rng.permuted(permutations, axis=1, out=permutations)
            n_permutations = permutations.shape[0]
            n_subsets = n_permutations * self.iteration_cost

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
                    interaction_index = self._interaction_lookup[interaction]
                    counts[interaction_index] += 1
                    for L in powerset(interaction):
                        game_value = game_values[subset_index]
                        update = game_value * (-1) ** (self.max_order - len(L))
                        result[interaction_index] += update
                        subset_index += 1

            used_budget += self.iteration_cost * batch_size

        # compute mean of interactions
        result = np.divide(result, counts, out=result, where=counts != 0)

        return self._finalize_result(result, budget=used_budget, estimated=True)

    def _compute_iteration_cost(self) -> int:
        """Computes the cost of performing a single iteration of the permutation sampling given
        the order, the number of players, and the STI index.

        Returns:
            int: The cost of a single iteration.
        """
        iteration_cost = int(binom(self.n, self.max_order) * 2**self.max_order)
        return iteration_cost

    def _compute_lower_order_sti(
        self, game: Callable[[np.ndarray], np.ndarray], result: np.ndarray[float]
    ) -> np.ndarray[float]:
        """Computes all lower order interactions for the STI index up to order max_order - 1.

        Args:
            game: The game function as a callable that takes a set of players and returns the value.
            result: The result array.

        Returns:
            The result array.
        """
        # get all game values on the whole powerset of players up to order max_order - 1
        lower_order_sizes = list(range(0, self.max_order))
        subsets: np.ndarray[bool] = get_explicit_subsets(self.n, lower_order_sizes)
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
                interaction_index = self._interaction_lookup[subset]
                result[interaction_index] += update
        return result
