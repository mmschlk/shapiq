"""This module contains the shapiq estimator."""
import math
from typing import Optional, Callable
import numpy as np

from approximator._base import Approximator, ShapleySamplingMixin, InteractionValues
from utils import powerset, split_subsets_budget

AVAILABLE_INDICES_SHAPIQ = {"SII, STI, FSI, nSII"}


class ShapIQ(Approximator, ShapleySamplingMixin):
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

    Properties:
        iteration_cost: The cost of a single iteration of the permutation sampling.

    Example:
        >>> from games import DummyGame
        >>> from approximator import ShapIQ
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = ShapIQ(n=5, max_order=2, index="SII")
        >>> approximator.approximate(budget=200, game=game)
        InteractionValues(
            index=SII, order=2, estimated=True, estimation_budget=200,
            values={
                1: [0.2 0.7 0.7 0.2 0.2]
                2: [[ 0.  0.  0.  0.  0.]
                    [ 0.  0.  1.  0.  0.]
                    [ 0.  0.  0.  0.  0.]
                    [ 0.  0.  0.  0.  0.]
                    [ 0.  0.  0.  0.  0.]]
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
        ShapleySamplingMixin.__init__(self)
        self._iteration_cost: int = 1
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
        estimation_flag = True

        # create storage array for given budget
        all_subsets: np.ndarray[bool] = np.zeros(shape=(budget, self.n), dtype=bool)

        # split the subset sizes into explicit and sampling parts
        sampling_weights: np.ndarray[float] = self._init_ksh_sampling_weights()
        explicit_sizes, sampling_sizes, remaining_budget = split_subsets_budget(
            order=1, n=self.n, budget=budget, sampling_weights=sampling_weights
        )

        # enumerate all explicit subsets
        explicit_subsets: np.ndarray[bool] = self._get_explicit_subsets(self.n, explicit_sizes)
        n_explicit_subsets = explicit_subsets.shape[0]
        all_subsets[:n_explicit_subsets] = explicit_subsets
        sampling_weights[explicit_sizes] = 0.0  # zero out sampling weights for explicit sizes

        # sample the remaining subsets with the remaining budget
        if len(sampling_sizes) > 0:
            if remaining_budget > 0:
                sampling_subsets: np.ndarray[bool] = self._sample_subsets(
                    budget=remaining_budget,
                    sampling_weights=sampling_weights,
                    replacement=replacement,
                    pairing=pairing,
                )
                all_subsets[len(explicit_subsets) :] = sampling_subsets
                # TODO there is an error with broadcasting here
                # "ValueError: could not broadcast input array from shape (18,5) into shape (19,5)"
                # from games import DummyGame
                # game = DummyGame(n=5, interaction=(1, 2))
                # approximator = ShapIQ(n=5, max_order=2, index="SII")
                # approximator.approximate(budget=29, game=game)
        else:
            estimation_flag = False  # no sampling needed computation is exact
            all_subsets = all_subsets[:n_explicit_subsets]  # remove unnecessary rows

        # add empty and full set to all_subsets in the beginning
        all_subsets = np.concatenate(
            (
                np.zeros(shape=(1, self.n), dtype=bool),  # empty set
                np.ones(shape=(1, self.n), dtype=bool),  # full set
                all_subsets,  # explicit and sampled subsets
            )
        )
        n_subsets = all_subsets.shape[0]
        n_explicit_subsets += 2  # add empty and full set

        n_iterations, last_batch_size = self._get_n_iterations(
            n_subsets, batch_size, iteration_cost=self._iteration_cost
        )

        # main computation loop
        result_explicit = self._init_result()
        result_sampled = self._init_result()
        counts = self._init_result(dtype=int)
        n_evaluated = 0
        for iteration in range(1, n_iterations + 1):
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
                    if subset_is_explicit:
                        result_explicit[interaction_size][interaction] += update
                    else:
                        result_sampled[interaction_size][interaction] += update
                        counts[interaction_size][interaction] += 1

            used_budget += batch_size

        # combine explicit and sampled parts
        result = {}
        for order in self._order_iterator:
            result_sampled[order] = np.divide(
                result_sampled[order],
                counts[order],
                out=result_sampled[order],
                where=counts[order] != 0,
            )
            result[order] = result_explicit[order] + result_sampled[order]

        return self._finalize_result(result, budget=used_budget, estimated=estimation_flag)

    def _weight_kernel(self, subset_size: int, interaction_size: int) -> float:
        """Returns the weight for each interaction type for a subset of size t and interaction of
        size s.

        Args:
            subset_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        t, s = subset_size, interaction_size
        if self.index == "SII" or self.index == "nSII":
            return (
                math.factorial(self.n - t - s) * math.factorial(t) / math.factorial(self.n - s + 1)
            )
        if self.index == "STI":
            if s == self.max_order:
                return (
                    self.max_order
                    * math.factorial(self.n - t - 1)
                    * math.factorial(t)
                    / math.factorial(self.n)
                )
            else:
                return 1.0 * (t == 0)
        if self.index == "FSI":
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

    def _init_discrete_derivative_weights(self) -> dict[int, np.ndarray[float]]:
        """Initializes the discrete derivative weights which are specific to each interaction index.

        Returns:
            The discrete derivative update weights.
        """
        # init data structure
        weights = {}
        for order in range(self.min_order, self.max_order + 1):
            weights[order] = np.zeros((self.n + 1, order + 1))
        # fill with values specific to each index
        for t in range(0, self.n + 1):
            for order in range(self.min_order, self.max_order + 1):
                for k in range(max(0, order + t - self.n), min(order, t) + 1):
                    weights[order][t, k] = (-1) ** (order - k) * self._weight_kernel(t - k, order)
        return weights


if __name__ == "__main__":
    from games import DummyGame

    game = DummyGame(n=5, interaction=(1, 2))
    approximator = ShapIQ(n=5, max_order=2, index="SII")
    print(approximator.approximate(budget=29, game=game))
