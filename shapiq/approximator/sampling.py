"""This module contains stochastic sampling procedures for coalitions of players."""

from __future__ import annotations

import copy
import warnings

import numpy as np
from scipy.special import binom

from shapiq.utils.sets import powerset


class CoalitionSampler:
    """Coalition Sampler for handling coalition sampling in approximation methods.

    The coalition sampler to generate a collection of subsets as a basis for approximation
    methods. Sampling is based on a more general variant of `Fumagalli et al. (2023) <https://doi.org/10.48550/arXiv.2303.01179>`_.
    The empty and grand coalition are always prioritized, and sampling budget is required ``>=2``.
    All variables are stored in the sampler, no objects are returned. The following variables
    are computed:
        - ``sampled_coalitions_matrix``: A binary matrix that consists of one row for each sampled
            coalition. Each row is a binary vector that indicates the players in the coalition.
            The matrix is of shape ``(n_coalitions, n_players)``.
        - ``sampled_coalitions_counter``: An array with the number of occurrences of the coalitions
            in the sampling process. The array is of shape ``(n_coalitions,)``.
        - ``sampled_coalitions_probability``: An array with the coalition probabilities according to
            the sampling procedure (i.e., the sampling weights). The array is of shape
            ``(n_coalitions,)``.
        - ``coalitions_per_size``: An array with the number of sampled coalitions per size
            (including the empty and full set). The array is of shape ``(n_players + 1,)``.
        - ``is_coalition_size_sampled``: An array that contains True, if the coalition size was
            sampled and False (computed exactly) otherwise. The array is of shape
            ``(n_players + 1,)``.
        - ``sampled_coalitions_dict``:`` A dictionary containing all sampled coalitions mapping to
            their number of occurrences. The dictionary is of type ``dict[tuple[int, ...], int]``.

    Attributes:
        n: The number of players in the game.

        n_max_coalitions: The maximum number of possible coalitions.

        adjusted_sampling_weights: The adjusted sampling weights without zero-weighted coalition sizes.
            The array is of shape ``(n_sizes_to_sample,)``.

        _rng: The random number generator used for sampling.


    Properties:
        sampled: A flag indicating whether the sampling process has been executed.

        coalitions_matrix: The binary matrix of sampled coalitions of shape ``(n_coalitions,
            n_players)``.

        coalitions_counter: The number of occurrences of the coalitions. The array is of shape
            ``(n_coalitions,)``.

        coalitions_probability: The coalition probabilities according to the sampling procedure. The
             array is of shape ``(n_coalitions,)``.

        coalitions_size_probability: The coalitions size probabilities according to the sampling
            procedure. The array is of shape ``(n_coalitions,)``.

        coalitions_size_probability: The coalitions probabilities in their size according to the
            sampling procedure. The array is of shape ``(n_coalitions,)``.

        n_coalitions: The number of coalitions that have been sampled.

        sampling_adjustment_weights: The weights that account for the sampling procedure (importance
            sampling)

        sampling_size_probabilities: The probabilities of each coalition size to be sampled.

    Examples:
        >>> sampler = CoalitionSampler(n_players=3, sampling_weights=np.array([1, 0.5, 0.5, 1]))
        >>> sampler.sample(5)
        >>> print(sampler.coalitions_matrix)
        [[False, False, False],
         [False, False, True],
         [True, True, True],
         [True, False, False],
         [False, True, True]]

    """

    def __init__(
        self,
        n_players: int,
        sampling_weights: np.ndarray,
        *,
        pairing_trick: bool = False,
        replacement: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the coalition sampler.

        Args:
            n_players: The number of players in the game.

            sampling_weights: Sampling for weights for coalition sizes, must be non-negative and at
                least one ``>0``. The sampling weights for size ``0`` and ``n`` are ignored, as
                these are always sampled.

            pairing_trick: Samples each coalition jointly with its complement. Defaults to
                ``False``.
            flag_replacement: If ``True``, the sampling is done with replacement

            random_state: The random state to use for the sampling process. Defaults to ``None``.
        """
        self.pairing_trick: bool = pairing_trick
        self.replacement: bool = replacement

        # set sampling weights
        if not (
            sampling_weights >= 0
        ).all():  # Check non-negativity of sampling weights
            msg = "All sampling weights must be non-negative"
            raise ValueError(msg)
        self._sampling_weights = sampling_weights / np.sum(
            sampling_weights
        )  # make probabilities

        # raise warning if sampling weights are not symmetric but pairing trick is activated
        if self.pairing_trick and not np.allclose(
            self._sampling_weights,
            self._sampling_weights[::-1],
        ):
            warnings.warn(
                UserWarning(
                    "Pairing trick is activated, but sampling weights are not symmetric. "
                    "This may lead to unexpected results.",
                ),
                stacklevel=2,
            )

        # set player numbers
        if n_players + 1 != np.size(
            sampling_weights
        ):  # shape of sampling weights -> sizes 0,...,n
            msg = (
                f"{n_players} elements must correspond to {n_players + 1} coalition sizes "
                "(including empty subsets)"
            )
            raise ValueError(msg)
        self.n: int = n_players
        self.n_max_coalitions = int(2**self.n)
        self.n_max_coalitions_per_size = np.array(
            [binom(self.n, k) for k in range(self.n + 1)]
        )

        # set random state
        self._rng: np.random.Generator = np.random.default_rng(seed=random_state)

        # set variables for sampling and exclude coalition sizes with zero weight
        self._coalitions_to_exclude: list[int] = []
        for size, weight in enumerate(self._sampling_weights):
            if weight == 0 and 0 < size < self.n:
                self.n_max_coalitions -= int(binom(self.n, size))
                self._coalitions_to_exclude.extend([size])
        self.adjusted_sampling_weights: np.ndarray[float] | None = None

        # set sample size variables (for border trick)
        self._coalitions_to_compute: list[int] | None = None  # coalitions to compute
        self._coalitions_to_sample: list[int] | None = None  # coalitions to sample

        # initialize variables to be computed and stored
        self.sampled_coalitions_dict: dict[
            tuple[int, ...], int
        ] | None = None  # coal -> count
        self.coalitions_per_size: np.ndarray[
            int
        ] | None = None  # number of coalitions per size

        # variables accessible through properties
        self._sampled_coalitions_matrix: np.ndarray[bool] | None = None  # coalitions
        self._sampled_coalitions_counter: np.ndarray[
            int
        ] | None = None  # coalitions_counter
        self._sampled_coalitions_size_prob: np.ndarray[
            float
        ] | None = None  # coalitions_size_probability
        self._sampled_coalitions_in_size_prob: np.ndarray[
            float
        ] | None = None  # coalitions_in_size_probability
        self._is_coalition_size_sampled: np.ndarray[
            bool
        ] | None = None  # is_coalition_size_sampled

    @property
    def n_coalitions(self) -> int:
        """Returns the number of coalitions that have been sampled.

        Returns:
            The number of coalitions that have been sampled.

        """
        try:
            return int(self._sampled_coalitions_matrix.shape[0])
        except AttributeError:  # if not sampled
            return 0

    @property
    def is_coalition_size_sampled(self) -> np.ndarray:
        """Returns a Boolean array indicating whether the coalition size was sampled.

        Returns:
            The Boolean array whether the coalition size was sampled.

        """
        return copy.deepcopy(self._is_coalition_size_sampled)

    @property
    def is_coalition_sampled(self) -> np.ndarray:
        """Returns a Boolean array indicating whether the coalition was sampled.

        Returns:
            The Boolean array whether the coalition was sampled.

        """
        coalitions_size = np.sum(self.coalitions_matrix, axis=1)
        return self._is_coalition_size_sampled[coalitions_size]

    @property
    def sampling_adjustment_weights(self) -> np.ndarray:
        """Returns the weights that account for the sampling procedure.

        Returns:
            An array with adjusted weight for each coalition

        """
        coalitions_counter = self.coalitions_counter
        is_coalition_sampled = self.is_coalition_sampled
        # Number of coalitions sampled

        n_total_samples = np.sum(coalitions_counter[is_coalition_sampled])
        # Helper array for computed and sampled coalitions
        total_samples_values = np.array([1, n_total_samples])
        # Create array per coalition and the total samples values, or 1, if computed
        n_coalitions_total_samples = total_samples_values[
            is_coalition_sampled.astype(int)
        ]
        # Create array with the adjusted weights
        return self.coalitions_counter / (
            self.coalitions_probability * n_coalitions_total_samples
        )

    @property
    def coalitions_matrix(self) -> np.ndarray:
        """Returns the binary matrix of sampled coalitions.

        Returns:
            A copy of the sampled coalitions matrix as a binary matrix of shape (n_coalitions,
                n_players).

        """
        return copy.deepcopy(self._sampled_coalitions_matrix)

    @property
    def sampling_size_probabilities(self) -> np.ndarray:
        """Returns the probabilities of sampling a coalition size.

        Returns:
            An array containing the probabilities of shappe ``(n+1,)``

        """
        size_probs = np.zeros(self.n + 1)
        size_probs[
            self._coalitions_to_sample
        ] = self.adjusted_sampling_weights / np.sum(
            self.adjusted_sampling_weights,
        )
        return size_probs

    @property
    def coalitions_counter(self) -> np.ndarray:
        """Returns the number of occurrences of the coalitions.

        Returns:
            A copy of the sampled coalitions counter of shape ``(n_coalitions,)``.

        """
        return copy.deepcopy(self._sampled_coalitions_counter)

    @property
    def coalitions_probability(self) -> np.ndarray | None:
        """Returns the coalition probabilities according to the sampling procedure.

        Returns the coalition probabilities according to the sampling procedure. The coalitions'
        probability is calculated as the product of the probability of the size of the coalition
        times the probability of the coalition in that size.

        Returns:
            A copy of the sampled coalitions probabilities of shape ``(n_coalitions,)`` or ``None``
                if the coalition probabilities are not available.

        """
        if (
            self._sampled_coalitions_size_prob is not None
            and self._sampled_coalitions_in_size_prob is not None
        ):
            return (
                self._sampled_coalitions_size_prob
                * self._sampled_coalitions_in_size_prob
            )
        return None

    @property
    def coalitions_size_probability(self) -> np.ndarray:
        """Returns the probabilities of the coalition sizes according to the sampling procedure.

        Returns:
            A copy of the probabilities of shape (n_coalitions,).

        """
        return copy.deepcopy(self._sampled_coalitions_size_prob)

    @property
    def coalitions_in_size_probability(self) -> np.ndarray:
        """Return probabilities per coalition size.

        Returns the probabilities of the coalition in the corresponding coalition size according
        to the sampling.

        Note:
            With uniform sampling, this is always ``1/binom(n,coalition_size)``.

        Returns:
            A copy of the sampled probabilities of shape ``(n_coalitions,)``.

        """
        return copy.deepcopy(self._sampled_coalitions_in_size_prob)

    @property
    def coalitions_size(self) -> np.ndarray:
        """Returns the coalition sizes of the sampled coalitions.

        Returns:
            The coalition sizes of the sampled coalitions.

        """
        return np.sum(self.coalitions_matrix, axis=1)

    @property
    def empty_coalition_index(self) -> int | None:
        """Returns the index of the empty coalition.

        Returns:
            The index of the empty coalition or ``None`` if the empty coalition was not sampled.

        """
        try:
            if self.coalitions_per_size[0] >= 1:
                return int(np.where(self.coalitions_size == 0)[0][0])
        except TypeError:
            pass
        return None

    def set_random_state(self, random_state: int | None = None) -> None:
        """Set the random state for the coalition sampler.

        Args:
            random_state: The random state to set. If ``None``, no random state is set. Defaults to
                ``None``.

        """
        self._rng = np.random.default_rng(seed=random_state)

    def execute_border_trick(self, sampling_budget: int) -> int:
        """Execute the border trick for a sampling budget.

        Moves coalition sizes from coalitions_to_sample to coalitions_to_compute, if the expected
        number of coalitions is higher than the total number of coalitions of that size. The border
        trick is based on a more general version of `Fumagalli et al. (2023) <https://doi.org/10.48550/arXiv.2303.01179>`_.

        Args:
            sampling_budget: The number of coalitions to sample.

        Returns:
            The sampling budget reduced by the number of coalitions in ``coalitions_to_compute``.

        """

        coalitions_per_size = np.array([binom(self.n, k) for k in range(self.n + 1)])
        expected_number_of_coalitions = sampling_budget * self.adjusted_sampling_weights
        sampling_exceeds_expectation = (
            expected_number_of_coalitions
            >= coalitions_per_size[self._coalitions_to_sample]
        )
        while sampling_exceeds_expectation.any():
            coalitions_to_move = [
                self._coalitions_to_sample[index]
                for index, include in enumerate(sampling_exceeds_expectation)
                if include
            ]
            self._coalitions_to_compute.extend(
                [
                    self._coalitions_to_sample.pop(
                        self._coalitions_to_sample.index(move_this)
                    )
                    for move_this in coalitions_to_move
                ],
            )
            sampling_budget -= int(np.sum(coalitions_per_size[coalitions_to_move]))
            self.adjusted_sampling_weights = self.adjusted_sampling_weights[
                ~sampling_exceeds_expectation
            ] / np.sum(self.adjusted_sampling_weights[~sampling_exceeds_expectation])
            expected_number_of_coalitions = (
                sampling_budget * self.adjusted_sampling_weights
            )
            sampling_exceeds_expectation = (
                expected_number_of_coalitions
                >= coalitions_per_size[self._coalitions_to_sample]
            )
        return sampling_budget

    def execute_pairing_trick(
        self, sampling_budget: int, coalition_tuple: tuple[int, ...]
    ) -> int:
        """Executes the pairing-trick for a sampling budget and coalition sizes.

        The pairing-trick is based on the idea by `Covert and Lee (2021) <https://doi.org/10.48550/arXiv.2012.01536>`_
        and pairs each coalition with its complement.

        Args:
            sampling_budget: The currently remaining sampling budget.
            coalition_tuple: The coalition to pair with its complement.

        Returns:
            The remaining sampling budget after the pairing-trick.

        """
        coalition_size = len(coalition_tuple)
        paired_coalition_size = self.n - coalition_size
        if paired_coalition_size in self._coalitions_to_sample:
            paired_coalition_indices = list(set(range(self.n)) - set(coalition_tuple))
            paired_coalition_tuple = tuple(sorted(paired_coalition_indices))
            sampling_budget -= self.add_coalition(paired_coalition_tuple)
        return sampling_budget

    def add_coalition(self, coalition_tuple):
        """Adds a sample to self.sampled_coalitions_dict based on replacement option

        Returns the new sampling budget and a Boolean indicating whether the coalition was added.

        """
        # add coalition
        if self.replacement:
            # if sampling with replacement, we can sample the same coalition multiple times
            self.coalitions_per_size[len(coalition_tuple)] += 1
            try:  # if coalition is not new
                self.sampled_coalitions_dict[coalition_tuple] += 1
                sampling_budget_spent = 0
            except KeyError:  # if coalition is new
                self.sampled_coalitions_dict[coalition_tuple] = 1
                sampling_budget_spent = 1
        # if sampling without replacement, we can only sample each coalition once
        elif coalition_tuple not in self.sampled_coalitions_dict:
            self.sampled_coalitions_dict[coalition_tuple] = 1
            self.coalitions_per_size[len(coalition_tuple)] += 1
            sampling_budget_spent = 1
        else:
            sampling_budget_spent = 0  # Coalition already seen
        return sampling_budget_spent

    def _reset_variables(self, sampling_budget: int) -> None:
        """Resets the variables of the sampler at each sampling call.

        Args:
            sampling_budget: The budget for the approximation (i.e., the number of distinct
                coalitions to sample/evaluate).

        """
        self.sampled_coalitions_dict = {}
        self.coalitions_per_size = np.zeros(self.n + 1, dtype=int)
        self._is_coalition_size_sampled = np.zeros(self.n + 1, dtype=bool)
        self._sampled_coalitions_counter = np.zeros(sampling_budget, dtype=int)
        self._sampled_coalitions_matrix = np.zeros(
            (sampling_budget, self.n), dtype=bool
        )
        self._sampled_coalitions_size_prob = np.zeros(sampling_budget, dtype=float)
        self._sampled_coalitions_in_size_prob = np.zeros(sampling_budget, dtype=float)

        self._coalitions_to_compute = []
        self._coalitions_to_sample = [
            coalition_size
            for coalition_size in range(self.n + 1)
            if coalition_size not in self._coalitions_to_exclude
        ]
        self.adjusted_sampling_weights = copy.deepcopy(
            self._sampling_weights[self._coalitions_to_sample],
        )
        self.adjusted_sampling_weights /= np.sum(
            self.adjusted_sampling_weights
        )  # probability

    def execute_empty_grand_coalition(self, sampling_budget: int) -> int:
        """Sets the empty and grand coalition to be computed.

        Ensures empty and grand coalition are prioritized and computed independent of
        the sampling weights. Works similar to border-trick but only with empty and grand coalition.

        Args:
            sampling_budget: The budget for the approximation (i.e., the number of distinct
                coalitions to sample/evaluate).

        Returns:
            The remaining sampling budget, i.e. reduced by ``2``.

        """
        empty_grand_coalition_indicator = np.zeros_like(
            self.adjusted_sampling_weights, dtype=bool
        )
        empty_grand_coalition_size = [0, self.n]
        empty_grand_coalition_index = [
            self._coalitions_to_sample.index(size)
            for size in empty_grand_coalition_size
        ]
        empty_grand_coalition_indicator[empty_grand_coalition_index] = True
        coalitions_to_move = [
            self._coalitions_to_sample[index]
            for index, include in enumerate(empty_grand_coalition_indicator)
            if include
        ]
        self._coalitions_to_compute.extend(
            [
                self._coalitions_to_sample.pop(
                    self._coalitions_to_sample.index(move_this)
                )
                for move_this in coalitions_to_move
            ],
        )
        self.adjusted_sampling_weights = self.adjusted_sampling_weights[
            ~empty_grand_coalition_indicator
        ] / np.sum(self.adjusted_sampling_weights[~empty_grand_coalition_indicator])
        sampling_budget -= 2
        return sampling_budget

    def sample(self, sampling_budget: int) -> None:
        """Samples distinct coalitions according to the specified budget.

        The empty and grand coalition are always prioritized, and sampling budget is required ``>=2``.

        Args:
            sampling_budget: The budget for the approximation (i.e., the number of distinct
                coalitions to sample/evaluate).

        Raises:
            UserWarning: If the sampling budget is higher than the maximum number of coalitions.

        """
        if sampling_budget < 2:
            # Empty and grand coalition always have to be computed.
            msg = "A minimum sampling budget of 2 samples is required."
            raise ValueError(msg)

        if sampling_budget > self.n_max_coalitions:
            warnings.warn(
                "Not all budget is required due to the border-trick.", stacklevel=2
            )
            sampling_budget = min(
                sampling_budget, self.n_max_coalitions
            )  # set budget to max coals

        self._reset_variables(sampling_budget)

        # Prioritize empty and grand coalition
        sampling_budget = self.execute_empty_grand_coalition(sampling_budget)

        # Border-Trick: enumerate all coalitions, where the expected number of coalitions exceeds
        # the total number of coalitions of that size (i.e. binom(n_players, coalition_size))
        sampling_budget = self.execute_border_trick(sampling_budget)

        # Sort by size for esthetics
        self._coalitions_to_compute.sort(key=self._sort_coalitions)

        # raise warning if budget is higher than 90% of samples remaining to be sampled
        n_samples_remaining = np.sum(
            [binom(self.n, size) for size in self._coalitions_to_sample]
        )
        if sampling_budget > 0.9 * n_samples_remaining:
            warnings.warn(
                UserWarning(
                    "Sampling might be inefficient (stalls) due to the sampling budget being close "
                    "to the total number of coalitions to be sampled.",
                ),
                stacklevel=2,
            )

        # sample coalitions
        if len(self._coalitions_to_sample) > 0:
            while sampling_budget > 0:
                # draw coalition
                coalition_size = self._rng.choice(
                    self._coalitions_to_sample,
                    size=1,
                    p=self.adjusted_sampling_weights,
                )[0]
                ids = self._rng.choice(self.n, size=coalition_size, replace=False)
                coalition_tuple = tuple(sorted(ids))  # get coalition

                sampling_budget -= self.add_coalition(coalition_tuple)
                # execute pairing-trick by including the complement
                if self.pairing_trick and sampling_budget > 0:
                    sampling_budget = self.execute_pairing_trick(
                        sampling_budget, coalition_tuple
                    )

        # convert coalition counts to the output format
        coalition_index = 0
        # add all coalitions that are computed exhaustively
        for coalition_size in self._coalitions_to_compute:
            self.coalitions_per_size[coalition_size] = int(
                binom(self.n, coalition_size)
            )
            for coalition in powerset(
                range(self.n),
                min_size=coalition_size,
                max_size=coalition_size,
            ):
                self._sampled_coalitions_matrix[coalition_index, list(coalition)] = 1
                self._sampled_coalitions_counter[coalition_index] = 1
                self._sampled_coalitions_size_prob[
                    coalition_index
                ] = 1  # weight is set to 1
                self._sampled_coalitions_in_size_prob[
                    coalition_index
                ] = 1  # weight is set to 1
                coalition_index += 1
        # add all coalitions that are sampled
        for coalition_tuple, count in self.sampled_coalitions_dict.items():
            self._sampled_coalitions_matrix[coalition_index, list(coalition_tuple)] = 1
            self._sampled_coalitions_counter[coalition_index] = count
            # probability of the sampled coalition, i.e. sampling weight (for size) divided by
            # number of coalitions of that size
            self._sampled_coalitions_size_prob[
                coalition_index
            ] = self.adjusted_sampling_weights[
                self._coalitions_to_sample.index(len(coalition_tuple))
            ]
            self._sampled_coalitions_in_size_prob[coalition_index] = (
                1 / self.n_max_coalitions_per_size[len(coalition_tuple)]
            )
            coalition_index += 1

        # set the flag to indicate that these sizes are sampled
        for coalition_size in self._coalitions_to_sample:
            self._is_coalition_size_sampled[coalition_size] = True

    def _sort_coalitions(self, value: int) -> float:
        """Used to sort coalition sizes by distance to center, i.e. grand coalition and emptyset first.

        Args:
            value: The size of the coalition.

        Returns:
            The negative distance to the center n/2

        """
        # Sort by distance to center
        return -abs(self.n / 2 - value)

# Helpers for fast sampler
import math
import itertools

def symmetric_round_even(x):
    # Function written by ChatGPT
    '''
    tgt is the nearest even integer to sum(x)
    Rounds the entries of x to integers so that the sum is tgt, and x is symmetric
    '''
    x = np.asarray(x, float); n = x.size
    tgt = int(np.round(x.sum()/2)*2)           # nearest even ≤ sum
    out = np.floor(x).astype(int)
    rem = tgt - out.sum()
    frac = x - np.floor(x)

    pairs = [(i, n-1-i, frac[i]+frac[n-1-i]) for i in range(n//2)]
    pairs.sort(key=lambda t: t[2], reverse=True)
    for i, j, _ in pairs:
        if rem < 2: break
        out[i] += 1; out[j] += 1; rem -= 2
    if n % 2 == 1 and rem == 1:                # give lone +1 to the center
        out[n//2] += 1; rem -= 1
    return out

from typing import Sequence, Tuple, TypeVar

T = TypeVar("T")

def ith_combination(pool: Sequence[T], size: int, index: int) -> Tuple[T, ...]:
    """
    Return the `index`-th k-combination of `pool` (0-based), in lexicographic
    order w.r.t. `pool`'s current order. Single pass, no while-loop.
    """
    n = len(pool)
    k = size

    if not (0 <= k <= n):
        raise ValueError("size must be between 0 and len(pool)")
    total = math.comb(n, k)
    if not (0 <= index < total):
        raise IndexError(f"index must be in [0, {total-1}] for C({n},{k})")

    combo = []
    for i in range(n):
        if k == 0:
            break

        # If we must take all remaining items
        if n - i == k:
            combo.extend(pool[i:i+k])
            k = 0
            break

        # Combinations that start by taking pool[i]
        c = math.comb(n - i - 1, k - 1)

        if index < c:
            combo.append(pool[i])
            k -= 1
        else:
            index -= c

    return tuple(combo)

def combination_generator(gen, n, s, num_samples):
    """
    Generate num_samples random combinations of s elements from a pool num_samples of size n in two settings:
    1. If the number of combinations is small (converting to an int does NOT cause an overflow error), randomly sample num_samples integers without replacement and generate the corresponding combinations on the fly with ith_combination.
    2. If the number of combinations is large (converting to an int DOES cause an overflow error), randomly sample num_samples combinations directly with replacement.
    """
    num_combos = math.comb(n, s)
    try:
        indices = gen.choice(num_combos, num_samples, replace=False)
        for i in indices:
            yield ith_combination(range(n), s, i)
    except OverflowError:
        for _ in range(num_samples):
            yield gen.choice(n, s, replace=False)

class CoalitionSamplerFast:
    '''
    Based on the sampling procedure by Musco and Witter (2025): https://arxiv.org/abs/2410.01917
    Code based on their implementation, with a few tweaks: https://github.com/rtealwitter/leverageshap/blob/7f4aa7c3417d02e5e71f3b86c28072cb2e560d30/leverageshap/estimators/regression.py
    '''
    def __init__(
        self,
        n_players: int,
        sampling_weights: np.ndarray,
        *,
        pairing_trick: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_players = n_players
        self.distribution = sampling_weights / np.min(sampling_weights)
        # Ensure smallest weight is 1
        self.pairing_trick = pairing_trick
        self.rng = np.random.default_rng(seed=random_state)
    
    def sampling_probs(self, sizes):
        return np.minimum(
            self.constant * self.distribution[sizes] / binom(self.n_players, sizes), 1
        )
    
    def get_sampling_probs(self, budget: int):
        # Function written by ChatGPT
        """
        Compute sampling probabilities without iteration by inverting the
        piecewise-linear function:
            E(c) = sum_k min(c * weights[k], comb_counts[k])
        where comb_counts[k] = C(n_players, k) and weights[k] = distribution[k].

        For any budget in [0, 2**n_players], this solves for a scale c such that
        E(c) ~= budget (up to floating-point error) and returns sampling_probs(sizes).
        """
        n = self.n_players
        sizes = np.arange(n + 1)

        # Per-size caps = number of coalitions of that size
        comb_counts = binom(n, sizes).astype(float)          # C(n, k)
        # Per-size weights from the distribution (>= 1 by construction)
        weights = self.distribution[sizes].astype(float)

        # Target expected total, clipped to feasible range [0, 2^n]
        target_total = float(np.clip(budget, 0, np.sum(comb_counts)))
        if target_total == 0.0:
            self.constant = 0.0
            return self.sampling_probs(sizes)

        # Breakpoints where a term saturates: c >= comb_counts[k] / weights[k]
        saturation_thresholds = comb_counts / weights
        order = np.argsort(saturation_thresholds)
        comb_counts_sorted = comb_counts[order]
        weights_sorted = weights[order]
        thresholds_sorted = saturation_thresholds[order]

        # For the segment before saturating index k:
        #   E(c) = sum_{j<k} comb_counts_sorted[j] + c * sum_{j>=k} weights_sorted[j]
        saturated_prefix = np.concatenate(([0.0], np.cumsum(comb_counts_sorted[:-1])))
        weights_prefix = np.concatenate(([0.0], np.cumsum(weights_sorted[:-1])))
        remaining_weight = np.sum(weights_sorted) - weights_prefix

        # Expected total at each breakpoint (just as k would start saturating)
        expected_at_threshold = saturated_prefix + thresholds_sorted * remaining_weight

        # Find the first segment where target_total fits
        segment_idx = np.searchsorted(expected_at_threshold, target_total, side="left")

        if segment_idx >= len(thresholds_sorted):
            # Past all segments: all terms saturate
            scale = float(thresholds_sorted[-1])
        else:
            denom = remaining_weight[segment_idx]
            # If denom == 0, slope is zero (nothing left to grow) -> stick to the threshold
            scale = thresholds_sorted[segment_idx] if denom == 0 else \
                    min((target_total - saturated_prefix[segment_idx]) / denom,
                        thresholds_sorted[segment_idx])

        self.constant = float(scale)
    
    def add_one_sample(self, indices):
        self.coalitions_matrix[self._coalition_idx, indices] = 1
        self.sampled_coalitions_dict[tuple(sorted(indices))] = 1
        self._coalition_idx += 1 

    def sample(self, budget: int):
        # Budget is an EVEN number between 2 and 2^n
        assert budget >= 2, "Budget must be at least 2"
        budget = min(budget, 2**self.n_players)
        budget += budget % 2  # make even

        # Get sampling probabilities
        self.get_sampling_probs(budget)
        sizes = np.arange(self.n_players + 1)
        samples_per_size = symmetric_round_even(
            self.sampling_probs(sizes) * binom(self.n_players, sizes)
        )
        sampling_probs = samples_per_size / binom(self.n_players, sizes)

        # Initialize storage
        num_samples = np.sum(samples_per_size)
        self.coalitions_matrix = np.zeros((num_samples, self.n_players), dtype=bool)
        self._coalition_idx = 0
        self.sampled_coalitions_dict = {}

        # Sample coalitions
        for idx, size in enumerate(sizes):
            if samples_per_size[idx] == math.comb(self.n_players, size):
                combo_gen = itertools.combinations(range(self.n_players), size)
                for indices in combo_gen:
                    self.add_one_sample(list(indices))
            elif self.pairing_trick and size == self.n_players // 2 and self.n_players % 2 == 0:
                combo_gen = combination_generator(
                    self.rng, self.n_players - 1, size - 1, samples_per_size[idx] // 2
                )
                for indices in combo_gen:
                    self.add_one_sample(list(indices) + [self.n_players - 1])
                    self.add_one_sample(list(set(range(self.n_players)) - set(indices)))
            else:
                combo_gen = combination_generator(
                    self.rng, self.n_players, size, samples_per_size[idx]
                )
                for indices in combo_gen:
                    self.add_one_sample(list(indices))
                    if self.pairing_trick:
                        self.add_one_sample(
                            list(set(range(self.n_players)) - set(indices))
                        )

        coalition_sizes = np.sum(self.coalitions_matrix, axis=1)
        self.sampling_adjustment_weights = 1 / sampling_probs[coalition_sizes]
