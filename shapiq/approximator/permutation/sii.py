"""This module implements the Permutation Sampling approximator for the SII (and k-SII) index."""

from typing import Callable, Optional

import numpy as np
from approximator._base import Approximator
from approximator.k_sii import KShapleyMixin
from interaction_values import InteractionValues
from utils import powerset


class PermutationSamplingSII(Approximator, KShapleyMixin):
    """Permutation Sampling approximator for the SII (and k-SII) index.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        top_order: Whether to approximate only the top order interactions (`True`) or all orders up
            to the specified order (`False`).
        random_state: The random state to use for the permutation sampling. Defaults to `None`.

    Attributes:
        n: The number of players.
        max_order: The interaction order of the approximation.
        top_order: Whether to approximate only the top order interactions (`True`) or all orders up
            to the specified order (`False`).
        min_order: The minimum order to approximate.
        iteration_cost: The cost of a single iteration of the permutation sampling.

    Example:
        >>> from games import DummyGame
        >>> from approximator import PermutationSamplingSII
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = PermutationSamplingSII(n=5, max_order=2)
        >>> approximator.approximate(budget=1_000, game=game)
        InteractionValues(
            index=SII, order=2, estimated=True, estimation_budget=988,
            values={
                (0,): 0.2,
                (1,): 0.7,
                (2,): 0.7,
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

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str = "SII",
        top_order: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        if index not in ["SII", "k-SII"]:
            raise ValueError(f"Invalid index {index}. Must be either 'SII' or 'k-SII'.")
        super().__init__(n, max_order, index, top_order, random_state)
        self.iteration_cost: int = self._compute_iteration_cost()

    def _compute_iteration_cost(self) -> int:
        """Computes the cost of performing a single iteration of the permutation sampling given
        the order, the number of players, and the SII index.

        Returns:
            int: The cost of a single iteration.
        """
        iteration_cost: int = 0
        for s in self._order_iterator:
            iteration_cost += (self.n - s + 1) * 2**s
        return iteration_cost

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
        batch_size: Optional[int] = 5,
    ) -> InteractionValues:
        """Approximates the interaction values.

        Args:
            budget: The budget for the approximation.
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If None, the batch size is set to 1. Defaults to 5.

        Returns:
            InteractionValues: The estimated interaction values.
        """

        batch_size = 1 if batch_size is None else batch_size
        used_budget = 0

        result: np.ndarray[float] = self._init_result()
        counts: np.ndarray[int] = self._init_result(dtype=int)

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget, batch_size, self.iteration_cost
        )

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
                for order in self._order_iterator:
                    for k in range(self.n - order + 1):
                        subset = permutations[permutation_id, k : k + order]
                        previous_subset = permutations[permutation_id, :k]
                        for subset_ in powerset(subset, min_size=0):
                            subset_eval = np.concatenate((previous_subset, subset_)).astype(int)
                            subsets[subset_index, subset_eval] = True
                            subset_index += 1

            # evaluate all subsets on the game
            game_values: np.ndarray[float] = game(subsets)

            # update the interaction scores by iterating over the permutations again
            subset_index = 0
            for permutation_id in range(n_permutations):
                for order in self._order_iterator:
                    for k in range(self.n - order + 1):
                        interaction = permutations[permutation_id, k : k + order]
                        interaction = tuple(sorted(interaction))
                        interaction_index = self._interaction_lookup[interaction]
                        counts[interaction_index] += 1
                        # update the discrete derivative given the subset
                        for subset_ in powerset(interaction, min_size=0):
                            game_value = game_values[subset_index]
                            update = game_value * (-1) ** (order - len(subset_))
                            result[interaction_index] += update
                            subset_index += 1

            used_budget += self.iteration_cost * batch_size

        # compute mean of interactions
        result = np.divide(result, counts, out=result, where=counts != 0)

        if self.index == "k-SII":
            result: np.ndarray[float] = self.transforms_sii_to_ksii(result)

        return self._finalize_result(result, budget=used_budget, estimated=True)
