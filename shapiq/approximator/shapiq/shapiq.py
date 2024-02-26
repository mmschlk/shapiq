"""This module contains the shapiq estimator."""

import math
from typing import Callable, Optional

import numpy as np
from approximator._base import Approximator
from approximator.k_sii import KShapleyMixin
from approximator.sampling import ShapleySamplingMixin
from interaction_values import InteractionValues
from utils import powerset

AVAILABLE_INDICES_SHAPIQ = {"SII", "STI", "FSI", "k-SII"}


class ShapIQ(Approximator, ShapleySamplingMixin, KShapleyMixin):
    """The ShapIQ estimator.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The index to approximate. Defaults to "SII".
        top_order: Whether to approximate only the top order interactions (`True`) or all orders up
            to the specified order (`False`). Defaults to `False`.
        random_state: The random state to use for the permutation sampling. Defaults to `None`.

    Attributes:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The index to approximate.
        top_order: Whether to approximate only the top order interactions (`True`) or all orders up
            to the specified order (`False`).
        min_order: The minimum order to approximate.
        iteration_cost: The cost of a single iteration of the permutation sampling.

    Example:
        >>> from games import DummyGame
        >>> from approximator import ShapIQ
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = ShapIQ(n=5, max_order=2, index="SII")
        >>> approximator.approximate(budget=50, game=game)
        InteractionValues(
            index=SII, order=2, estimated=False, estimation_budget=32,
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
        super().__init__(
            n, max_order=max_order, index=index, top_order=top_order, random_state=random_state
        )
        self.iteration_cost: int = 1
        self._weights = self._init_discrete_derivative_weights()

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
        batch_size: int = None,
        pairing: bool = True,
        replacement: bool = True,
    ) -> InteractionValues:
        """Approximates the interaction values using the ShapIQ estimator.

        Args:
            budget: The budget for the approximation (i.e., the number of game evaluations).
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If None, the batch size is set to `budget`.
                Defaults to None.
            pairing: Whether to use pairwise sampling (`True`) or not (`False`). Defaults to `True`.
                Paired sampling can increase the approximation quality.
            replacement: Whether to sample with replacement (`True`) or without replacement
                (`False`). Defaults to `True`.

        Returns:
            The estimated interaction values.
        """
        # validate input parameters
        batch_size = budget + 2 if batch_size is None else batch_size
        used_budget = 0

        # generate the dataset containing explicit and sampled subsets
        all_subsets, estimation_flag, n_explicit_subsets = self._generate_shapley_dataset(
            budget, pairing, replacement
        )
        n_subsets = all_subsets.shape[0]

        # calculate the number of iterations and the last batch size
        n_iterations, last_batch_size = self._calc_iteration_count(
            n_subsets, batch_size, iteration_cost=self.iteration_cost
        )

        # main computation loop
        result_explicit: np.ndarray[float] = self._init_result()
        result_sampled: np.ndarray[float] = self._init_result()
        counts: np.ndarray[int] = self._init_result(dtype=int)
        n_evaluated = 0
        for iteration in range(1, n_iterations + 1):
            # get the batch of subsets and game evals
            batch_size = batch_size if iteration != n_iterations else last_batch_size
            batch_index = batch_size * (iteration - 1)
            batch_subsets = all_subsets[batch_index : batch_index + batch_size]
            game_values = game(batch_subsets)

            # update the interaction scores by iterating over the subsets in the batch
            for subset_index, subset in enumerate(batch_subsets):
                subset = sorted(tuple(np.where(subset)[0]))  # T
                n_evaluated += 1
                subset_is_explicit = subset_index < n_explicit_subsets
                subset_size = len(subset)  # |T|
                game_eval = game_values[subset_index]  # \nu(T)
                for interaction in powerset(self.N, self.min_order, self.max_order):
                    intersection_size = len(set(interaction).intersection(set(subset)))  # |S\capT|
                    interaction_size = len(interaction)  # |S|
                    update = (
                        game_eval * self._weights[interaction_size][subset_size, intersection_size]
                    )
                    interaction_index = self._interaction_lookup[interaction]
                    if subset_is_explicit:
                        result_explicit[interaction_index] += update
                    else:
                        result_sampled[interaction_index] += update
                        counts[interaction_index] += 1
            used_budget += batch_size

        # combine explicit and sampled parts
        result_sampled = np.divide(result_sampled, counts, out=result_sampled, where=counts != 0)
        result = result_explicit + result_sampled

        if self.index == "k-SII":
            result: np.ndarray[float] = self.transforms_sii_to_ksii(result)

        return self._finalize_result(result, budget=used_budget, estimated=estimation_flag)

    def _sii_weight_kernel(self, subset_size: int, interaction_size: int) -> float:
        """Returns the SII discrete derivative weight given the subset size and interaction size.

        TODO add formula and reference to paper.

        Args:
            subset_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        t, s = subset_size, interaction_size
        return math.factorial(self.n - t - s) * math.factorial(t) / math.factorial(self.n - s + 1)

    def _sti_weight_kernel(self, subset_size: int, interaction_size: int) -> float:
        """Returns the STI discrete derivative weight given the subset size and interaction size.

        TODO add formula and reference to paper.

        Args:
            subset_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        t, s = subset_size, interaction_size
        if s == self.max_order:
            return (
                self.max_order
                * math.factorial(self.n - t - 1)
                * math.factorial(t)
                / math.factorial(self.n)
            )
        else:
            return 1.0 * (t == 0)

    def _fsi_weight_kernel(self, subset_size: int, interaction_size: int) -> float:
        """Returns the FSI discrete derivative weight given the subset size and interaction size.

        TODO add formula and reference to paper.

        Args:
            subset_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        t, s = subset_size, interaction_size
        if s == self.max_order:
            return (
                math.factorial(2 * s - 1)
                / math.factorial(s - 1) ** 2
                * math.factorial(self.n - t - 1)
                * math.factorial(t + s - 1)
                / math.factorial(self.n + s - 1)
            )
        else:
            raise ValueError("Lower order interactions are not supported.")

    def _weight_kernel(self, subset_size: int, interaction_size: int) -> float:
        """Returns the weight for each interaction type for a subset of size t and interaction of
        size s.

        Args:
            subset_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if self.index == "SII" or self.index == "k-SII":  # in both cases return SII kernel
            return self._sii_weight_kernel(subset_size, interaction_size)
        elif self.index == "STI":
            return self._sti_weight_kernel(subset_size, interaction_size)
        elif self.index == "FSI":
            return self._fsi_weight_kernel(subset_size, interaction_size)
        else:
            raise ValueError(f"Unknown index {self.index}.")

    def _init_discrete_derivative_weights(self) -> dict[int, np.ndarray[float]]:
        """Initializes the discrete derivative weights which are specific to each interaction index.

        Returns:
            The discrete derivative update weights.
        """
        # init data structure
        weights = {}
        for order in self._order_iterator:
            weights[order] = np.zeros((self.n + 1, order + 1))
        # fill with values specific to each index
        for t in range(0, self.n + 1):
            for order in self._order_iterator:
                for k in range(max(0, order + t - self.n), min(order, t) + 1):
                    weights[order][t, k] = (-1) ** (order - k) * self._weight_kernel(t - k, order)
        return weights
