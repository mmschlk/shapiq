"""This module contains the base regression algorithms to estimate FSI scores."""
from typing import Optional, Callable, Union

import numpy as np
from scipy.special import binom

from approximator._base import Approximator, InteractionValues

AVAILABLE_INDICES_REGRESSION = {"FSI"}


class Regression(Approximator):
    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        random_state: Optional[int] = None,
    ) -> None:
        if index not in AVAILABLE_INDICES_REGRESSION:
            raise ValueError(
                f"Index {index} is not valid. "
                f"Available indices are {AVAILABLE_INDICES_REGRESSION}."
            )
        super().__init__(n, max_order, index, False, random_state)
        self._big_M = float(1_000_000)

    def approximate(
        self, budget: int, game: Callable[[Union[set, tuple]], float], *args, **kwargs
    ) -> InteractionValues:
        """Approximates the interaction values."""
        raise NotImplementedError

    def _init_ksh_sampling_weights(self) -> np.ndarray[float]:
        """Initializes the weights for sampling subsets.

        The sampling weights are of size n + 1 and indexed by the size of the subset. The edges
        (the first, empty coalition, and the last element, full coalition) are set to 0.

        Returns:
            The weights for sampling subsets.
        """
        weight_vector = np.zeros(shape=self.n - 1, dtype=float)
        for subset_size in range(1, self.n):
            weight_vector[subset_size - 1] = (self.n - 1) / (subset_size * (self.n - subset_size))
        sampling_weight = (np.asarray([0] + [*weight_vector] + [0])) / sum(weight_vector)
        return sampling_weight

    def _get_ksh_subset_weights(self, subsets: np.ndarray[bool]) -> np.ndarray[float]:
        """Computes the KernelSHAP regression weights for the given subsets.

        The weights for the subsets of size s are set to ksh_weights[s] / binom(n, s). The weights
        for the empty and full sets are set to a big number.

        Args:
            subsets: one-hot matrix of subsets for which to compute the weights in shape
                (n_subsets, n).

        Returns:
            The KernelSHAP regression weights in shape (n_subsets,).
        """
        # set the weights for each subset to ksh_weights[|S|] / binom(n, |S|)
        ksh_weights = self._init_ksh_sampling_weights()  # indexed by subset size
        subset_sizes = np.sum(subsets, axis=1)
        weights = ksh_weights[subset_sizes]  # set the weights for each subset size
        weights /= binom(self.n, subset_sizes)  # divide by the number of subsets of the same size

        # set the weights for the empty and full sets to big M
        weights[np.logical_not(subsets).all(axis=1)] = self._big_M
        weights[subsets.all(axis=1)] = self._big_M
        return weights
