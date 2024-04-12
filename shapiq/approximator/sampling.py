"""This module contains stochastic sampling procedures for coalitions of players."""

import copy
import warnings
from abc import ABC
from typing import Optional, Union

import numpy as np
import scipy as sp
from scipy.special import binom

from shapiq.approximator._base import Approximator
from shapiq.utils import get_explicit_subsets, split_subsets_budget
from shapiq.utils.sets import powerset


class CoalitionSampler:
    """The coalition sampler to generate a collection of subsets as a basis for approximation
    methods.

    Sampling is based on a more general variant of [Fumagalli et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html)
    All variables are stored in the sampler, no objects are returned. The following variables
    are computed:
        - sampled_coalitions_matrix: A binary matrix that consists of one row for each sampled
            coalition. Each row is a binary vector that indicates the players in the coalition.
            The matrix is of shape (n_coalitions, n_players).
        - sampled_coalitions_counter: An array with the number of occurrences of the coalitions
            in the sampling process. The array is of shape (n_coalitions,).
        - sampled_coalitions_probability: An array with the coalition probabilities according to the
            sampling procedure (i.e., the sampling weights). The array is of shape (n_coalitions,).
        - coalitions_per_size: An array with the number of sampled coalitions per size (including
            the empty and full set). The array is of shape (n_players + 1,).
        - is_coalition_size_sampled: An array that contains True, if the coalition size was
            sampled and False (computed exactly) otherwise. The array is of shape (n_players + 1,).
        - sampled_coalitions_dict: A dictionary containing all sampled coalitions mapping to their
            number of occurrences. The dictionary is of type dict[tuple[int, ...], int].

    Args:
        n_players: The number of players in the game.
        sampling_weights: Sampling for weights for coalition sizes, must be non-negative and at least one >0.
        pairing_trick: Samples each coalition jointly with its complement, default is False
        random_state: The random state to use for the sampling process. Defaults to `None`.

    Attributes and Properties:
        n: The number of players in the game.
        n_max_coalitions: The maximum number of possible coalitions.
        adjusted_sampling_weights: The adjusted sampling weights without zero-weighted coalition
            sizes. The array is of shape (n_sizes_to_sample,).
        sampled: A flag indicating whether the sampling process has been executed.
        coalitions_matrix: The binary matrix of sampled coalitions of shape (n_coalitions,
            n_players).
        coalitions_counter: The number of occurrences of the coalitions. The array is of shape
            (n_coalitions,).
        coalitions_probability: The coalition probabilities according to the sampling procedure. The
             array is of shape (n_coalitions,).
        n_coalitions: The number of coalitions that have been sampled.
    """

    def __init__(
        self,
        n_players: int,
        sampling_weights: np.ndarray,
        pairing_trick: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.pairing_trick: bool = pairing_trick

        # set sampling weights
        if not (sampling_weights >= 0).all():  # Check non-negativity of sampling weights
            raise ValueError("All sampling weights must be non-negative")
        self._sampling_weights = sampling_weights / np.sum(sampling_weights)  # make probabilities

        # raise warning if sampling weights are not symmetric but pairing trick is activated
        if self.pairing_trick and not np.allclose(
            self._sampling_weights, self._sampling_weights[::-1]
        ):
            warnings.warn(
                UserWarning(
                    "Pairing trick is activated, but sampling weights are not symmetric. "
                    "This may lead to unexpected results."
                )
            )

        # set player numbers
        if n_players + 1 != np.size(sampling_weights):  # shape of sampling weights -> sizes 0,...,n
            raise ValueError(
                f"{n_players} elements must correspond to {n_players + 1} coalition sizes "
                "(including empty subsets)"
            )
        self.n: int = n_players
        self.n_max_coalitions = int(2**self.n)

        # set random state
        self._rng: np.random.Generator = np.random.default_rng(seed=random_state)

        self._coalitions_to_compute: Optional[
            list
        ] = None  # coalitions to compute, set in sample method
        self._coalitions_to_sample: Optional[
            list
        ] = None  #  coalitions to sample, set in sample method

        # set variables for sampling
        self._coalitions_to_exclude = []

        # exclude coalition sizes with zero weight
        for size, weight in enumerate(self._sampling_weights):
            if weight == 0:
                self.n_max_coalitions -= int(binom(self.n, size))
                self._coalitions_to_exclude.extend([size])

        self.adjusted_sampling_weights: Optional[np.ndarray[float]] = None

        # initialize variables to be computed and stored
        self.sampled_coalitions_dict: Optional[dict[tuple[int, ...], int]] = None  # coal -> count
        self.coalitions_per_size: Optional[np.ndarray[int]] = None  # number of coalitions per size
        self.is_coalition_size_sampled: Optional[np.ndarray[bool]] = None  # flag if size is sampled

        # variables accessible through properties
        self._sampled_coalitions_matrix: Optional[np.ndarray[bool]] = None  # coalitions
        self._sampled_coalitions_counter: Optional[np.ndarray[int]] = None  # coalitions_counter
        self._sampled_coalitions_prob: Optional[np.ndarray[float]] = None  # coalitions_probability

        self.sampled = False

    @property
    def n_coalitions(self) -> int:
        """Returns the number of coalitions that have been sampled.

        Returns:
            The number of coalitions that have been sampled.
        """
        return int(self._sampled_coalitions_matrix.shape[0])

    @property
    def coalitions_matrix(self) -> np.ndarray:
        """Returns the binary matrix of sampled coalitions.

        Returns:
            A copy of the sampled coalitions matrix as a binary matrix of shape (n_coalitions,
                n_players).
        """
        return copy.deepcopy(self._sampled_coalitions_matrix)

    @property
    def coalitions_counter(self) -> np.ndarray:
        """Returns the number of occurrences of the coalitions

        Returns:
            A copy of the sampled coalitions counter of shape (n_coalitions,).
        """
        return copy.deepcopy(self._sampled_coalitions_counter)

    @property
    def coalitions_probability(self) -> np.ndarray:
        """Returns the coalition probabilities according to the sampling procedure.

        Returns:
            A copy of the sampled coalitions probabilities of shape (n_coalitions,).
        """
        return copy.deepcopy(self._sampled_coalitions_prob)

    def execute_border_trick(self, sampling_budget: int) -> int:
        """Moves coalition sizes from coalitions_to_sample to coalitions_to_compute, if the expected
        number of coalitions is higher than the total number of coalitions of that size.

        The border trick is based on a more general version of [Fumagalli et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html).

        Args:
            sampling_budget: The number of coalitions to sample

        Returns:
            The sampling budget reduced by the number of coalitions in coalitions_to_compute
        """
        coalitions_per_size = np.array([binom(self.n, k) for k in range(self.n + 1)])
        expected_number_of_coalitions = sampling_budget * self.adjusted_sampling_weights
        sampling_exceeds_expectation = (
            expected_number_of_coalitions >= coalitions_per_size[self._coalitions_to_sample]
        )
        while sampling_exceeds_expectation.any():
            coalitions_to_move = [
                self._coalitions_to_sample[index]
                for index, include in enumerate(sampling_exceeds_expectation)
                if include
            ]
            self._coalitions_to_compute.extend(
                [
                    self._coalitions_to_sample.pop(self._coalitions_to_sample.index(move_this))
                    for move_this in coalitions_to_move
                ]
            )
            sampling_budget -= int(np.sum(coalitions_per_size[coalitions_to_move]))
            self.adjusted_sampling_weights = self.adjusted_sampling_weights[
                ~sampling_exceeds_expectation
            ] / np.sum(self.adjusted_sampling_weights[~sampling_exceeds_expectation])
            expected_number_of_coalitions = sampling_budget * self.adjusted_sampling_weights
            sampling_exceeds_expectation = (
                expected_number_of_coalitions >= coalitions_per_size[self._coalitions_to_sample]
            )
        return sampling_budget

    def execute_pairing_trick(self, sampling_budget: int, coalition_size: int, permutation) -> int:
        """Executes the pairing-trick for a sampling budget and coalition sizes.

        The pairing-trick is based on the idea of
        [Covert and Lee 2021](https://proceedings.mlr.press/v130/covert21a.html) and pairs each
        coalition with its complement.

        Works similar as the initial coalition, but throws a warning, if the subset is not allowed
            for sampling.

        Args:
            sampling_budget: The currently remaining sampling budget.
            coalition_size: The coalition size of the coalition that should be paired.
            permutation: The permutation from which the coalition was drawn.

        Returns:
            The remaining sampling budget after the pairing-trick.
        """
        paired_coalition_size = self.n - coalition_size
        if paired_coalition_size in self._coalitions_to_sample:
            paired_coalition_indices = permutation[coalition_size:]
            paired_coalition_tuple = tuple(sorted(paired_coalition_indices))
            self.coalitions_per_size[paired_coalition_size] += 1
            # adjust coalitions counter using the paired coalition
            try:  # if coalition is not new
                self.sampled_coalitions_dict[paired_coalition_tuple] += 1
            except KeyError:  # if coalition is new
                self.sampled_coalitions_dict[paired_coalition_tuple] = 1
                sampling_budget -= 1
        return sampling_budget

    def _reset_variables(self, sampling_budget: int) -> None:
        """Resets the variables of the sampler at each sampling call.

        Args:
            sampling_budget: The budget for the approximation (i.e., the number of distinct
                coalitions to sample/evaluate).
        """
        self.sampled_coalitions_dict = {}
        self.coalitions_per_size = np.zeros(self.n + 1, dtype=int)
        self.is_coalition_size_sampled = np.zeros(self.n + 1, dtype=bool)
        self._sampled_coalitions_counter = np.zeros(sampling_budget, dtype=int)
        self._sampled_coalitions_matrix = np.zeros((sampling_budget, self.n), dtype=bool)
        self._sampled_coalitions_prob = np.zeros(sampling_budget, dtype=float)

        self._coalitions_to_compute = []
        self._coalitions_to_sample = [
            coalition_size
            for coalition_size in range(self.n + 1)
            if coalition_size not in self._coalitions_to_exclude
        ]
        self.adjusted_sampling_weights = copy.deepcopy(
            self._sampling_weights[self._coalitions_to_sample]
        )
        self.adjusted_sampling_weights /= np.sum(self.adjusted_sampling_weights)  # probability

    def sample(self, sampling_budget: int) -> None:
        """Samples distinct coalitions according to the specified budget.

        Args:
            sampling_budget: The budget for the approximation (i.e., the number of distinct
                coalitions to sample/evaluate).

        Raises:
            UserWarning: If the sampling budget is higher than the maximum number of coalitions.
        """

        if sampling_budget > self.n_max_coalitions:
            warnings.warn(UserWarning("Not all budget is required due to the border-trick."))
            sampling_budget = min(sampling_budget, self.n_max_coalitions)  # set budget to max coals

        self._reset_variables(sampling_budget)

        # Border-Trick: enumerate all coalitions, where the expected number of coalitions exceeds
        # the total number of coalitions of that size (i.e. binom(n_players, coalition_size))
        sampling_budget = self.execute_border_trick(sampling_budget)

        # Sort by size for esthetics
        self._coalitions_to_sample.sort()
        self._coalitions_to_compute.sort()

        # raise warning if budget is higher than 90% of samples remaining to be sampled
        n_samples_remaining = np.sum([binom(self.n, size) for size in self._coalitions_to_sample])
        if sampling_budget > 0.9 * n_samples_remaining:
            warnings.warn(
                UserWarning(
                    "Sampling might be inefficient (stalls) due to the sampling budget being close "
                    "to the total number of coalitions to be sampled."
                )
            )

        # sample coalitions
        if len(self._coalitions_to_sample) > 0:
            iteration_counter = 0  # stores the number of samples drawn (duplicates included)
            coalition_sizes, permutations = self._draw_coalition_sizes(n_draws=sampling_budget * 2)
            coalition_counter = 0  # stores the index of the coalition size

            while sampling_budget > 0:
                iteration_counter += 1

                # draw new coalition sizes and permutations if all are used
                if iteration_counter % len(coalition_sizes) == 0:
                    coalition_sizes, permutations = self._draw_coalition_sizes(sampling_budget)
                    coalition_counter = 0

                # extract coalition by size from permutation
                coalition_size = int(coalition_sizes[coalition_counter])  # get coalition size
                permutation = permutations[coalition_counter]  # get permutation for coalition
                coalition_tuple = tuple(sorted(permutation[:coalition_size]))  # get coalition
                self.coalitions_per_size[coalition_size] += 1

                # add coalition
                try:  # if coalition is not new
                    self.sampled_coalitions_dict[coalition_tuple] += 1
                except KeyError:  # if coalition is new
                    self.sampled_coalitions_dict[coalition_tuple] = 1
                    sampling_budget -= 1

                # execute pairing-trick by including the complement
                if self.pairing_trick and sampling_budget > 0:
                    sampling_budget = self.execute_pairing_trick(
                        sampling_budget, coalition_size, permutation
                    )

        # convert coalition counts to the output format
        coalition_index = 0
        # add all coalitions that are computed exhaustively
        for coalition_size in self._coalitions_to_compute:
            self.coalitions_per_size[coalition_size] = int(binom(self.n, coalition_size))
            for coalition in powerset(
                range(self.n), min_size=coalition_size, max_size=coalition_size
            ):
                self._sampled_coalitions_matrix[coalition_index, list(coalition)] = 1
                self._sampled_coalitions_counter[coalition_index] = 1
                self._sampled_coalitions_prob[coalition_index] = 1  # weight is set to 1
                coalition_index += 1
        # add all coalitions that are sampled
        for coalition_tuple, count in self.sampled_coalitions_dict.items():
            self._sampled_coalitions_matrix[coalition_index, list(coalition_tuple)] = 1
            self._sampled_coalitions_counter[coalition_index] = count
            # probability of the sampled coalition, i.e. sampling weight (for size) divided by
            # number of coalitions of that size
            self._sampled_coalitions_prob[coalition_index] = (
                self.adjusted_sampling_weights[
                    self._coalitions_to_sample.index(len(coalition_tuple))
                ]
                / self.coalitions_per_size[len(coalition_tuple)]
            )
            coalition_index += 1

        # set the flag to indicate that these sizes are sampled
        for coalition_size in self._coalitions_to_sample:
            self.is_coalition_size_sampled[coalition_size] = True

        self.sampled = True

    def _draw_coalition_sizes(self, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        """Draws coalition sizes for the sampling process.

        Args:
            n_draws: The number of coalition sizes to draw.

        Returns:
            The coalition sizes and permutations for the sampling process.
        """
        coalition_sizes = self._rng.choice(
            self._coalitions_to_sample,
            p=self.adjusted_sampling_weights,
            size=n_draws,
        )
        permutations = np.tile(np.arange(self.n, dtype=int), (n_draws, 1))
        self._rng.permuted(permutations, axis=1, out=permutations)
        return coalition_sizes, permutations


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
