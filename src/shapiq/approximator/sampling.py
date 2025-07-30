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

            random_state: The random state to use for the sampling process. Defaults to ``None``.
        """
        self.pairing_trick: bool = pairing_trick

        # set sampling weights
        if not (sampling_weights >= 0).all():  # Check non-negativity of sampling weights
            msg = "All sampling weights must be non-negative"
            raise ValueError(msg)
        self._sampling_weights = sampling_weights / np.sum(sampling_weights)  # make probabilities

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
        if n_players + 1 != np.size(sampling_weights):  # shape of sampling weights -> sizes 0,...,n
            msg = (
                f"{n_players} elements must correspond to {n_players + 1} coalition sizes "
                "(including empty subsets)"
            )
            raise ValueError(msg)
        self.n: int = n_players
        self.n_max_coalitions = int(2**self.n)
        self.n_max_coalitions_per_size = np.array([binom(self.n, k) for k in range(self.n + 1)])

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
        self.sampled_coalitions_dict: dict[tuple[int, ...], int] | None = None  # coal -> count
        self.coalitions_per_size: np.ndarray[int] | None = None  # number of coalitions per size

        # variables accessible through properties
        self._sampled_coalitions_matrix: np.ndarray[bool] | None = None  # coalitions
        self._sampled_coalitions_counter: np.ndarray[int] | None = None  # coalitions_counter
        self._sampled_coalitions_size_prob: np.ndarray[float] | None = (
            None  # coalitions_size_probability
        )
        self._sampled_coalitions_in_size_prob: np.ndarray[float] | None = (
            None  # coalitions_in_size_probability
        )
        self._is_coalition_size_sampled: np.ndarray[bool] | None = None  # is_coalition_size_sampled

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
        n_coalitions_total_samples = total_samples_values[is_coalition_sampled.astype(int)]
        # Create array with the adjusted weights
        return self.coalitions_counter / (self.coalitions_probability * n_coalitions_total_samples)

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
        size_probs[self._coalitions_to_sample] = self.adjusted_sampling_weights / np.sum(
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
            return self._sampled_coalitions_size_prob * self._sampled_coalitions_in_size_prob
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
                ],
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

    def execute_pairing_trick(self, sampling_budget: int, coalition_tuple: tuple[int, ...]) -> int:
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
        self._is_coalition_size_sampled = np.zeros(self.n + 1, dtype=bool)
        self._sampled_coalitions_counter = np.zeros(sampling_budget, dtype=int)
        self._sampled_coalitions_matrix = np.zeros((sampling_budget, self.n), dtype=bool)
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
        self.adjusted_sampling_weights /= np.sum(self.adjusted_sampling_weights)  # probability

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
        empty_grand_coalition_indicator = np.zeros_like(self.adjusted_sampling_weights, dtype=bool)
        empty_grand_coalition_size = [0, self.n]
        empty_grand_coalition_index = [
            self._coalitions_to_sample.index(size) for size in empty_grand_coalition_size
        ]
        empty_grand_coalition_indicator[empty_grand_coalition_index] = True
        coalitions_to_move = [
            self._coalitions_to_sample[index]
            for index, include in enumerate(empty_grand_coalition_indicator)
            if include
        ]
        self._coalitions_to_compute.extend(
            [
                self._coalitions_to_sample.pop(self._coalitions_to_sample.index(move_this))
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
            warnings.warn("Not all budget is required due to the border-trick.", stacklevel=2)
            sampling_budget = min(sampling_budget, self.n_max_coalitions)  # set budget to max coals

        self._reset_variables(sampling_budget)

        # Prioritize empty and grand coalition
        sampling_budget = self.execute_empty_grand_coalition(sampling_budget)

        # Border-Trick: enumerate all coalitions, where the expected number of coalitions exceeds
        # the total number of coalitions of that size (i.e. binom(n_players, coalition_size))
        sampling_budget = self.execute_border_trick(sampling_budget)

        # Sort by size for esthetics
        self._coalitions_to_compute.sort(key=self._sort_coalitions)

        # raise warning if budget is higher than 90% of samples remaining to be sampled
        n_samples_remaining = np.sum([binom(self.n, size) for size in self._coalitions_to_sample])
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
            iteration_counter = 0  # stores the number of samples drawn (duplicates included)
            while sampling_budget > 0:
                iteration_counter += 1

                # draw coalition
                coalition_size = self._rng.choice(
                    self._coalitions_to_sample,
                    size=1,
                    p=self.adjusted_sampling_weights,
                )[0]
                ids = self._rng.choice(self.n, size=coalition_size, replace=False)
                coalition_tuple = tuple(sorted(ids))  # get coalition
                self.coalitions_per_size[coalition_size] += 1

                # add coalition
                try:  # if coalition is not new
                    self.sampled_coalitions_dict[coalition_tuple] += 1
                except KeyError:  # if coalition is new
                    self.sampled_coalitions_dict[coalition_tuple] = 1
                    sampling_budget -= 1

                # execute pairing-trick by including the complement
                if self.pairing_trick and sampling_budget > 0:
                    sampling_budget = self.execute_pairing_trick(sampling_budget, coalition_tuple)

        # convert coalition counts to the output format
        coalition_index = 0
        # add all coalitions that are computed exhaustively
        for coalition_size in self._coalitions_to_compute:
            self.coalitions_per_size[coalition_size] = int(binom(self.n, coalition_size))
            for coalition in powerset(
                range(self.n),
                min_size=coalition_size,
                max_size=coalition_size,
            ):
                self._sampled_coalitions_matrix[coalition_index, list(coalition)] = 1
                self._sampled_coalitions_counter[coalition_index] = 1
                self._sampled_coalitions_size_prob[coalition_index] = 1  # weight is set to 1
                self._sampled_coalitions_in_size_prob[coalition_index] = 1  # weight is set to 1
                coalition_index += 1
        # add all coalitions that are sampled
        for coalition_tuple, count in self.sampled_coalitions_dict.items():
            self._sampled_coalitions_matrix[coalition_index, list(coalition_tuple)] = 1
            self._sampled_coalitions_counter[coalition_index] = count
            # probability of the sampled coalition, i.e. sampling weight (for size) divided by
            # number of coalitions of that size
            self._sampled_coalitions_size_prob[coalition_index] = self.adjusted_sampling_weights[
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
