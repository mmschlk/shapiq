import copy
from abc import ABC
from typing import Union

import numpy as np
from approximator._base import Approximator
from scipy.special import binom

from shapiq.utils import get_explicit_subsets, split_subsets_budget


class ShapleySamplingMixin(ABC):
    """Mixin class for the computation of Shapley weights.

    Provides the common functionality for regression-based approximators like
    :class:`~shapiq.approximators.RegressionFSI`. The class offers computation of Shapley weights
    and the corresponding sampling weights for the KernelSHAP-like estimation approaches.
    """

    def _init_ksh_sampling_weights(
        self: Union[Approximator, "ShapleySamplingMixin"]
    ) -> np.ndarray[float]:
        """Initializes the weights for sampling subsets.

        The sampling weights are of size n + 1 and indexed by the size of the subset. The edges
        (the first, empty coalition, and the last element, full coalition) are set to 0.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """

        weight_vector = np.zeros(shape=self.n - 1, dtype=float)
        for subset_size in range(1, self.n):
            weight_vector[subset_size - 1] = (self.n - 1) / (subset_size * (self.n - subset_size))
        sampling_weight = (np.asarray([0] + [*weight_vector] + [0])) / sum(weight_vector)
        return sampling_weight

    def _get_ksh_subset_weights(
        self: Union[Approximator, "ShapleySamplingMixin"], subsets: np.ndarray[bool]
    ) -> np.ndarray[float]:
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
        weights[np.logical_not(subsets).all(axis=1)] = float(1_000_000)
        weights[subsets.all(axis=1)] = float(1_000_000)
        return weights

    def _sample_subsets(
        self: Union[Approximator, "ShapleySamplingMixin"],
        budget: int,
        sampling_weights: np.ndarray[float],
        replacement: bool = False,
        pairing: bool = True,
    ) -> np.ndarray[bool]:
        """Samples subsets with the given budget.

        Args:
            budget: budget for the sampling.
            sampling_weights: weights for sampling subsets of certain sizes and indexed by the size.
                The shape is expected to be (n + 1,). A size that is not to be sampled has weight 0.
            pairing: whether to use pairing (`True`) sampling or not (`False`). Defaults to `False`.

        Returns:
            sampled subsets.
        """
        # sanitize input parameters
        sampling_weights = copy.copy(sampling_weights)
        sampling_weights /= np.sum(sampling_weights)

        # adjust budget for paired sampling
        if pairing:
            budget = budget - budget % 2  # must be even for pairing
            budget = int(budget / 2)

        # create storage array for given budget
        subset_matrix = np.zeros(shape=(budget, self.n), dtype=bool)

        # sample subsets
        sampled_sizes = self._rng.choice(self.N_arr, size=budget, p=sampling_weights).astype(int)
        if replacement:  # sample subsets with replacement
            permutations = np.tile(np.arange(self.n), (budget, 1))
            self._rng.permuted(permutations, axis=1, out=permutations)
            for i, subset_size in enumerate(sampled_sizes):
                subset = permutations[i, :subset_size]
                subset_matrix[i, subset] = True
        else:  # sample subsets without replacement
            sampled_subsets, n_sampled = set(), 0  # init sampling variables
            while n_sampled < budget:
                subset_size = sampled_sizes[n_sampled]
                subset = tuple(sorted(self._rng.choice(np.arange(0, self.n), size=subset_size)))
                sampled_subsets.add(subset)
                if len(sampled_subsets) != n_sampled:  # subset was not already sampled
                    subset_matrix[n_sampled, subset] = True
                    n_sampled += 1  # continue sampling

        if pairing:
            subset_matrix = np.repeat(subset_matrix, repeats=2, axis=0)  # extend the subset matrix
            subset_matrix[1::2] = np.logical_not(subset_matrix[1::2])  # flip sign of paired subsets

        return subset_matrix

    def _generate_shapley_dataset(
        self: Union[Approximator, "ShapleySamplingMixin"],
        budget: int,
        pairing: bool = True,
        replacement: bool = False,
    ) -> tuple[np.ndarray[bool], bool, int]:
        """Generates the two-part dataset containing explicit and sampled subsets.

        The first part of the dataset contains all explicit subsets. The second half contains the
        sampled subsets. The parts can be determined by the `n_explicit_subsets` parameter.

        Args:
            budget: The budget for the approximation (i.e., the number of allowed game evaluations).
            pairing: Whether to use pairwise sampling (`True`) or not (`False`). Defaults to `True`.
                Paired sampling can increase the approximation quality.
            replacement: Whether to sample with replacement (`True`) or without replacement
                (`False`). Defaults to `False`.

        Returns:
            - The dataset containing explicit and sampled subsets. The dataset is a 2D array of
                shape (n_subsets, n_players) where each row is a subset.
            - A flag indicating whether the approximation is estimated (`True`) or exact (`False`).
            - The number of explicit subsets.
        """
        estimation_flag = True
        # create storage array for given budget
        all_subsets: np.ndarray[bool] = np.zeros(shape=(budget, self.n), dtype=bool)
        n_subsets = 0
        # split the subset sizes into explicit and sampling parts
        sampling_weights: np.ndarray[float] = self._init_ksh_sampling_weights()
        explicit_sizes, sampling_sizes, remaining_budget = split_subsets_budget(
            order=1, n=self.n, budget=budget, sampling_weights=sampling_weights
        )
        # enumerate all explicit subsets
        explicit_subsets: np.ndarray[bool] = get_explicit_subsets(self.n, explicit_sizes)
        n_explicit_subsets = explicit_subsets.shape[0]
        all_subsets[:n_explicit_subsets] = explicit_subsets
        n_subsets += n_explicit_subsets
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
                n_subsets += sampling_subsets.shape[0]
                all_subsets[n_explicit_subsets:n_subsets] = sampling_subsets
                all_subsets = all_subsets[:n_subsets]  # remove unnecessary rows
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
        n_explicit_subsets += 2  # add empty and full set
        return all_subsets, estimation_flag, n_explicit_subsets
