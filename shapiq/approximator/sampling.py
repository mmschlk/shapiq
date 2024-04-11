"""This module contains stochastic sampling procedures for coalitions of players."""

# Used for CoalitionSampler
import copy
import warnings

# Used for ShapleySamplingMixin
from abc import ABC
from typing import Optional, Union

import numpy as np
import scipy as sp
from scipy.special import binom

from shapiq.approximator._base import Approximator
from shapiq.utils import get_explicit_subsets, split_subsets_budget
from shapiq.utils.sets import powerset


class CoalitionSampler:
    """The coalition sampler to generate a collection of subsets as a basis for approximation methods.

    Args:
        n: Number of elements to sample from
        sampling_weights: Sampling for weights for coalition sizes, must be non-negative and at least one >0.
        pairing_trick: Samples each coalition jointly with its complement, default is False
        random_state: The random state to use for the sampling process. Defaults to `None`.

    Attributes:
        budget: Sampling budget, i.e. number of distinct subset that will be sampled
        sampling_weights: Sampling for weights for coalition sizes.
        random_state: The random state to use for the sampling process. Defaults to `None`.
        n: Number of players
        N: Set of players
        pairing_trick: Boolean indicates whether pairing trick is activated
        coalitions_to_exclude: List of coalition sizes excluded from sampling due to zero weight
        coalitions_to_compute: List of coalition sizes exhaustively computed due to border-trick (changed in sample method)
        coalitions_to_sample: List of coalition sizes that are sampled (changed in sample method)
        n_max_coalitions: The maximum number of coalitions (excluding excluded coalition sizes)
        adjusted_sampling_weights: The adjusted sampling weights normalized for the current coalitions_to_sample list (chagned in sample method)
    Example:
    """

    def __init__(
        self,
        n: np.ndarray,
        sampling_weights: np.ndarray,
        pairing_trick: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.n = n
        self.N = np.array(range(n))
        self.sampling_weights = sampling_weights / np.sum(sampling_weights)
        self.pairing_trick = pairing_trick
        self._random_state = random_state
        self._rng: Optional[np.random.Generator] = np.random.default_rng(seed=self._random_state)
        self.coalitions_to_exclude = []
        self.coalitions_to_compute = []
        self.coalitions_to_sample = list(range(self.n + 1))
        # Set maximum of possible coalitions by excluding zero-weighted coalition sizes
        # Coalition sizes with zero weights, should be removed and excluded from total number of coalitions
        self.n_max_coalitions = int(2**self.n)
        # Handle zero-weighted coalition sizes, move to coalition_to_exclude
        for size, weight in enumerate(self.sampling_weights):
            if weight == 0:
                self.n_max_coalitions -= int(binom(self.n, size))
                self.coalitions_to_exclude.extend(
                    [self.coalitions_to_sample.pop(self.coalitions_to_sample.index(size))]
                )
        self.adjusted_sampling_weights = self.sampling_weights[self.coalitions_to_sample] / np.sum(
            self.sampling_weights[self.coalitions_to_sample]
        )
        # Check shape of sampling weights for coalition sizes 0,...,n
        assert n + 1 == np.size(
            sampling_weights
        ), "n elements must correspond to n+1 coalition sizes (including empty subsets)"
        # Check non-negativity of sampling weights
        assert (sampling_weights >= 0).all(), "All sampling weights must be non-negative"

    def get_coalitions_matrix(self):
        return copy.deepcopy(self.sampled_coalitions_matrix)

    def get_coalitions_counter(self):
        return copy.deepcopy(self.sampled_coalitions_counter)

    def get_coalitions_prob(self):
        return copy.deepcopy(self.sampled_coalitions_prob)

    def execute_border_trick(self, sampling_budget):
        """Moves coalition sizes from coalitions_to_sample to coalitions_to_compute,
        if the expected number of coalitions is higher than the total number of coalitions of that size.
        Based on a more general version of https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html

        Args:
            sampling_budget: The number of coalitions to sample

        Returns:
            The sampling budget reduced by the number of coalitions in coalitions_to_compute
        """
        coalitions_per_size = np.array([binom(self.n, k) for k in range(self.n + 1)])
        expected_number_of_coalitions = sampling_budget * self.adjusted_sampling_weights
        sampling_exceeds_expectation = (
            expected_number_of_coalitions >= coalitions_per_size[self.coalitions_to_sample]
        )
        while sampling_exceeds_expectation.any():
            coalitions_to_move = [
                self.coalitions_to_sample[index]
                for index, include in enumerate(sampling_exceeds_expectation)
                if include
            ]
            self.coalitions_to_compute.extend(
                [
                    self.coalitions_to_sample.pop(self.coalitions_to_sample.index(move_this))
                    for move_this in coalitions_to_move
                ]
            )
            sampling_budget -= int(np.sum(coalitions_per_size[coalitions_to_move]))
            self.adjusted_sampling_weights = self.adjusted_sampling_weights[
                ~sampling_exceeds_expectation
            ] / np.sum(self.adjusted_sampling_weights[~sampling_exceeds_expectation])
            expected_number_of_coalitions = sampling_budget * self.adjusted_sampling_weights
            sampling_exceeds_expectation = (
                expected_number_of_coalitions >= coalitions_per_size[self.coalitions_to_sample]
            )
        return sampling_budget

    def execute_pairing_trick(self, sampling_budget, coalition_size, permutation):
        paired_coalition_size = self.n - coalition_size
        if paired_coalition_size not in self.coalitions_to_sample:
            # If the coalition size of the complement is not sampled, throw warning
            warnings.warn(
                "Pairing is affected as weights are not symmetric.",
                UserWarning,
            )
        else:
            paired_coalition_indices = permutation[coalition_size:]
            paired_coalition_indices.sort()
            paired_coalition_tuple = tuple(paired_coalition_indices)
            self.coalitions_per_size[paired_coalition_size] += 1
            # Adjust coalitions counter using the paired coalition
            if self.sampled_coalitions_dict.get(paired_coalition_tuple):
                # if coalition is not new
                self.sampled_coalitions_dict[paired_coalition_tuple] += 1
            else:
                self.sampled_coalitions_dict[paired_coalition_tuple] = 1
                sampling_budget -= 1
        return sampling_budget

    def sample(self, sampling_budget: int) -> np.ndarray:
        """Samples distinct coalitions according to the specified budget.
        Sampling is based on a more general variant of https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html

        Args:
            budget: The budget for the approximation (i.e., the number of distinct coalitions).

        Returns:
            All variables are stored in the sampler, no objects are returned. The following variables are computed:
            - sampled_coalitions_matrix: A binary matrix that consists of one row for each sampled coalition
            - sampled_coalitions_counter: An array with the number of occurrences of the coalitions
            - sampled_coalitions_prob: An array with the coalition probabilities according to the sampling procedure
            - coalitions_per_size: An array with the number of sampled coalitions per size
            - is_coalition_size_sampled: An array that contains True, if the coalition size was sampled
            - coalitions_to_sample: The list of coalition sizes that are sampled
            - coalitions_to_compute: The list of coalition sizes that are exhaustively computed
            - sampled_coalitions_dict: A dictionary containing all sampled coalitions
        """

        self.sampled_coalitions_dict = {}

        if sampling_budget > self.n_max_coalitions:
            warnings.warn("Not all budget is required due to the border-trick.", UserWarning)
            # Adjust sampling budget to max coalitions
            sampling_budget = min(sampling_budget, self.n_max_coalitions)

        # Variables to be computed and stored
        self.coalitions_per_size = np.zeros(self.n + 1, dtype=int)
        self.sampled_coalitions_counter = np.zeros(sampling_budget)
        self.sampled_coalitions_matrix = np.zeros((sampling_budget, self.n), dtype=int)
        self.sampled_coalitions_prob = np.zeros((sampling_budget), dtype=float)
        self.is_coalition_size_sampled = np.zeros(self.n + 1, dtype=bool)
        permutation = np.arange(self.n)

        # Border-Trick: Enumerate all coalitions, where the expected number of coalitions exceeds the total number
        sampling_budget = self.execute_border_trick(sampling_budget)

        # Sort by size for esthetics
        self.coalitions_to_sample.sort()
        self.coalitions_to_compute.sort()
        if (
            len(self.coalitions_to_sample) > 0
        ):  # Sample, if there are coalition sizes to sample from
            # Set counter that stores the number of sampled coalitions (duplicates included)
            coalition_counter = 0
            # Start sampling procedure
            while sampling_budget > 0:
                if coalition_counter == 0:
                    # Draw random coalition sizes based on adjusted_sampling_weights if, repeat if necessary
                    coalition_sizes = self._rng.choice(
                        self.coalitions_to_sample,
                        p=self.adjusted_sampling_weights,
                        size=sampling_budget,
                    )
                    coalition_sizes_size = len(coalition_sizes)
                # generate random permutation
                self._rng.shuffle(permutation)

                coalition_size = coalition_sizes[coalition_counter]
                coalition_counter += 1
                # Reset counter to draw new coalition sizes, if all have been used
                coalition_counter %= coalition_sizes_size

                # Extract coalition by size from permutation
                coalition_indices = permutation[:coalition_size]
                coalition_indices.sort()  # Sorting for consistent representation
                coalition_tuple = tuple(coalition_indices)
                self.coalitions_per_size[coalition_size] += 1

                # Adjust coalitions counter
                if self.sampled_coalitions_dict.get(coalition_tuple):
                    # if coalition is not new
                    self.sampled_coalitions_dict[coalition_tuple] += 1
                else:
                    self.sampled_coalitions_dict[coalition_tuple] = 1
                    sampling_budget -= 1

                # Execute pairing-trick by including the complement, i.e. the rest of the permutation as a coalition
                if self.pairing_trick and sampling_budget > 0:
                    self.execute_pairing_trick(sampling_budget, coalition_size, permutation)

        # Convert coalition counts to a binary matrix
        coalition_index = 0
        # Add all coalitions that are computed exhaustively
        for coalition_size in self.coalitions_to_compute:
            self.coalitions_per_size[coalition_size] = int(binom(self.n, coalition_size))
            for coalition in powerset(
                range(self.n), min_size=coalition_size, max_size=coalition_size
            ):
                self.sampled_coalitions_matrix[coalition_index, list(coalition)] = 1
                self.sampled_coalitions_counter[coalition_index] = 1
                # Weight is set to 1
                self.sampled_coalitions_prob[coalition_index] = 1
                coalition_index += 1
        # Add all coalitions that are sampled
        for coalition_tuple, count in self.sampled_coalitions_dict.items():
            self.sampled_coalitions_matrix[coalition_index, list(coalition_tuple)] = 1
            self.sampled_coalitions_counter[coalition_index] = count
            # Probability of the sampled coalition,
            # i.e. sampling weight (for size) divided by number of coalitions of that size
            self.sampled_coalitions_prob[coalition_index] = (
                self.adjusted_sampling_weights[
                    self.coalitions_to_sample.index(len(coalition_tuple))
                ]
                / self.coalitions_per_size[len(coalition_tuple)]
            )
            coalition_index += 1

        for coalition_size in self.coalitions_to_sample:
            self.is_coalition_size_sampled[coalition_size] = True


class ShapleySamplingMixin(ABC):
    """Mixin class for the computation of Shapley weights.

    Provides the common functionality for regression-based approximators like
    :class:`~shapiq.approximators.RegressionFSI`. The class offers computation of Shapley weights
    and the corresponding sampling weights for the KernelSHAP-like estimation approaches.
    """

    def _init_ksh_sampling_weights(
        self: Union[Approximator, "ShapleySamplingMixin"],
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
        weights /= sp.special.binom(
            self.n, subset_sizes
        )  # divide by the number of subsets of the same size

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
