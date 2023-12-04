"""This module contains the base approximator classes for the shapiq package."""
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from scipy.special import binom, bernoulli
from utils import get_explicit_subsets, powerset, split_subsets_budget

AVAILABLE_INDICES = {"SII", "nSII", "STI", "FSI", "SV"}


__all__ = [
    "InteractionValues",
    "Approximator",
    "ShapleySamplingMixin",
    "NShapleyMixin",
    "transforms_sii_to_nsii",
    "convert_nsii_into_one_dimension",
]


@dataclass
class InteractionValues:
    """This class contains the interaction values as estimated by an approximator.

    Attributes:
        values: The interaction values of the model in vectorized form.
        index: The interaction index estimated. Available indices are 'SII', 'nSII', 'STI', and
            'FSI'.
        max_order: The order of the approximation.
        min_order: The minimum order of the approximation.
        n_players: The number of players.
        interaction_lookup: A dictionary that maps interactions to their index in the values
            vector. If `interaction_lookup` is not provided, it is computed from the `n_players`,
            `min_order`, and `max_order` parameters. Defaults to `None`.
        estimated: Whether the interaction values are estimated or not. Defaults to `True`.
        estimation_budget: The budget used for the estimation. Defaults to `None`.
    """

    values: np.ndarray[float]
    index: str
    max_order: int
    min_order: int
    n_players: int
    interaction_lookup: dict[tuple[int], int] = None
    estimated: bool = True
    estimation_budget: Optional[int] = None

    def __post_init__(self) -> None:
        """Checks if the index is valid."""
        if self.index not in AVAILABLE_INDICES:
            raise ValueError(
                f"Index {self.index} is not valid. " f"Available indices are {AVAILABLE_INDICES}."
            )
        if self.interaction_lookup is None:
            self.interaction_lookup = _generate_interaction_lookup(
                self.n_players, self.min_order, self.max_order
            )

    def __repr__(self) -> str:
        """Returns the representation of the InteractionValues object."""
        representation = "InteractionValues(\n"
        representation += (
            f"    index={self.index}, max_order={self.max_order}, min_order={self.min_order}"
            f", estimated={self.estimated}, estimation_budget={self.estimation_budget},\n"
        ) + "    values={\n"
        for interaction in powerset(
            set(range(self.n_players)), min_size=1, max_size=self.max_order
        ):
            representation += f"        {interaction}: "
            interaction_value = str(round(self[interaction], 4))
            interaction_value = interaction_value.replace("-0.0", "0.0").replace(" 0.0", " 0")
            interaction_value = interaction_value.replace("0.0 ", "0 ")
            representation += f"{interaction_value},\n"
        representation = representation[:-2]  # remove last "," and add closing bracket
        representation += "\n    }\n)"
        return representation

    def __str__(self) -> str:
        """Returns the string representation of the InteractionValues object."""
        return self.__repr__()

    def __getitem__(self, item: tuple[int, ...]) -> float:
        """Returns the score for the given interaction.

        Args:
            item: The interaction for which to return the score.

        Returns:
            The interaction value.
        """
        item = tuple(sorted(item))
        return float(self.values[self.interaction_lookup[item]])

    def __eq__(self, other: object) -> bool:
        """Checks if two InteractionValues objects are equal.

        Args:
            other: The other InteractionValues object.

        Returns:
            True if the two objects are equal, False otherwise.
        """
        if not isinstance(other, InteractionValues):
            raise TypeError("Cannot compare InteractionValues with other types.")
        if (
            self.index != other.index
            or self.max_order != other.max_order
            or self.n_players != other.n_players
        ):
            return False
        if not np.allclose(self.values, other.values):
            return False
        return True

    def __ne__(self, other: object) -> bool:
        """Checks if two InteractionValues objects are not equal.

        Args:
            other: The other InteractionValues object.

        Returns:
            True if the two objects are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Returns the hash of the InteractionValues object."""
        return hash((self.index, self.max_order, tuple(self.values.flatten())))

    def __copy__(self) -> "InteractionValues":
        """Returns a copy of the InteractionValues object."""
        return InteractionValues(
            values=copy.deepcopy(self.values),
            index=self.index,
            max_order=self.max_order,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
            n_players=self.n_players,
            interaction_lookup=copy.deepcopy(self.interaction_lookup),
            min_order=self.min_order,
        )

    def __deepcopy__(self, memo) -> "InteractionValues":
        """Returns a deep copy of the InteractionValues object."""
        return InteractionValues(
            values=copy.deepcopy(self.values),
            index=self.index,
            max_order=self.max_order,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
            n_players=self.n_players,
            interaction_lookup=copy.deepcopy(self.interaction_lookup),
            min_order=self.min_order,
        )


class Approximator(ABC):
    """This class is the base class for all approximators.

    Approximators are used to estimate the interaction values of a model or any value function.
    Different approximators can be used to estimate different interaction indices. Some can be used
    to estimate all indices.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated. Available indices are 'SII', 'nSII', 'STI',
            and 'FSI'.
        top_order: If True, the approximation is performed only for the top order interactions. If
            False, the approximation is performed for all orders up to the specified order.
        random_state: The random state to use for the approximation. Defaults to None.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        N_arr: The array of players (starting from 0 to n).
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated.
        top_order: If True, the approximation is performed only for the top order interactions. If
            False, the approximation is performed for all orders up to the specified order.
        min_order: The minimum order of the approximation. If top_order is True, min_order is equal
            to max_order. Otherwise, min_order is equal to 1.
        iteration_cost: The cost of a single iteration of the approximator.

    """

    @abstractmethod
    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        top_order: bool,
        random_state: Optional[int] = None,
    ) -> None:
        """Initializes the approximator."""
        self.index: str = index
        if self.index not in AVAILABLE_INDICES:
            raise ValueError(
                f"Index {self.index} is not valid. " f"Available indices are {AVAILABLE_INDICES}."
            )
        self.n: int = n
        self.N: set = set(range(self.n))
        self.N_arr: np.ndarray[int] = np.arange(self.n + 1)
        self.top_order: bool = top_order
        self.max_order: int = max_order
        self.min_order: int = self.max_order if self.top_order else 1
        self.iteration_cost: int = 1  # default value, can be overwritten by subclasses
        self._interaction_lookup = _generate_interaction_lookup(
            self.n, self.min_order, self.max_order
        )
        self._random_state: Optional[int] = random_state
        self._rng: Optional[np.random.Generator] = np.random.default_rng(seed=self._random_state)

    @abstractmethod
    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], *args, **kwargs
    ) -> InteractionValues:
        """Approximates the interaction values. Abstract method that needs to be implemented for
        each approximator.

        Args:
            budget: The budget for the approximation.
            game: The game function.

        Returns:
            The interaction values.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("The approximate method needs to be implemented.")

    def _init_result(self, dtype=float) -> np.ndarray:
        """Initializes the result array. The result array is a 1D array of size n_interactions as
        determined by the interaction_lookup dictionary.

        Args:
            dtype: The data type of the result array. Defaults to float.

        Returns:
            The result array.
        """
        result = np.zeros(len(self._interaction_lookup), dtype=dtype)
        return result

    @property
    def _order_iterator(self) -> range:
        """Returns an iterator over the orders of the approximation.

        Returns:
            The iterator.
        """
        return range(self.min_order, self.max_order + 1)

    def _finalize_result(
        self,
        result,
        estimated: bool = True,
        budget: Optional[int] = None,
        index: Optional[str] = None,
    ) -> InteractionValues:
        """Finalizes the result dictionary.

        Args:
            result: The result dictionary.
            estimated: Whether the interaction values are estimated or not. Defaults to True.
            budget: The budget used for the estimation. Defaults to None.
            index: The interaction index estimated. Available indices are 'SII', 'nSII', 'STI', and
                'FSI'. Defaults to None (i.e., the index of the approximator is used).

        Returns:
            The interaction values.
        """
        if index is None:
            index = self.index
        return InteractionValues(
            values=result,
            estimated=estimated,
            estimation_budget=budget,
            index=index,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup=self._interaction_lookup,
        )

    @staticmethod
    def _calc_iteration_count(budget: int, batch_size: int, iteration_cost: int) -> tuple[int, int]:
        """Computes the number of iterations and the size of the last batch given the batch size and
        the budget.

        Args:
            budget: The budget for the approximation.
            batch_size: The size of the batch.
            iteration_cost: The cost of a single iteration.

        Returns:
            int, int: The number of iterations and the size of the last batch.
        """
        n_iterations = budget // (iteration_cost * batch_size)
        last_batch_size = batch_size
        remaining_budget = budget - n_iterations * iteration_cost * batch_size
        if remaining_budget > 0 and remaining_budget // iteration_cost > 0:
            last_batch_size = remaining_budget // iteration_cost
            n_iterations += 1
        return n_iterations, last_batch_size

    def __repr__(self) -> str:
        """Returns the representation of the Approximator object."""
        return (
            f"{self.__class__.__name__}(\n"
            f"    n={self.n},\n"
            f"    max_order={self.max_order},\n"
            f"    index={self.index},\n"
            f"    top_order={self.top_order},\n"
            f"    random_state={self._random_state}\n"
            f")"
        )

    def __str__(self) -> str:
        """Returns the string representation of the Approximator object."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Checks if two Approximator objects are equal."""
        if not isinstance(other, Approximator):
            raise ValueError("Cannot compare Approximator with other types.")
        if (
            self.n != other.n
            or self.max_order != other.max_order
            or self.index != other.index
            or self.top_order != other.top_order
            or self._random_state != other._random_state
        ):
            return False
        return True

    def __ne__(self, other: object) -> bool:
        """Checks if two Approximator objects are not equal."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Returns the hash of the Approximator object."""
        return hash((self.n, self.max_order, self.index, self.top_order, self._random_state))

    @property
    def interaction_lookup(self):
        return self._interaction_lookup


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


class NShapleyMixin:
    """Mixin class for the computation of n-Shapley values from SII estimators.

    Provides the common functionality for SII-based approximators like `PermutationSamplingSII` or
    `ShapIQ` for SII to transform their interaction scores into nSII values. The nSII values are
    proposed in this `paper<https://proceedings.mlr.press/v206/bordt23a>`_.
    """

    def transforms_sii_to_nsii(
        self: Approximator,
        sii_values: Union[np.ndarray[float], InteractionValues],
    ) -> Union[np.ndarray[float], InteractionValues]:
        """Transforms the SII values into nSII values.

        Args:
            sii_values: The SII values to transform. Can be either a numpy array or an
                InteractionValues object. The output will be of the same type.

        Returns:
            The nSII values in the same format as the input.
        """
        return transforms_sii_to_nsii(
            sii_values=sii_values,
            approximator=self,
        )


def transforms_sii_to_nsii(
    sii_values: Union[np.ndarray[float], InteractionValues],
    *,
    approximator: Optional[Approximator] = None,
    n: Optional[int] = None,
    max_order: Optional[int] = None,
    interaction_lookup: Optional[dict] = None,
) -> Union[np.ndarray[float], InteractionValues]:
    """Transforms the SII values into nSII values.

    Args:
        sii_values: The SII values to transform. Can be either a numpy array or an
            InteractionValues object. The output will be of the same type.
        approximator: The approximator used to estimate the SII values. If provided, meta
            information for the transformation is taken from the approximator. Defaults to None.
        n: The number of players. Required if `approximator` is not provided. Defaults to None.
        max_order: The maximum order of the approximation. Required if `approximator` is not
            provided. Defaults to None.
        interaction_lookup: A dictionary that maps interactions to their index in the values
            vector. If `interaction_lookup` is not provided, it is computed from the `n_players`
            and the `max_order` parameters. Defaults to `None`.

    Returns:
        The nSII values in the same format as the input.
    """
    if isinstance(sii_values, InteractionValues):
        n_sii_values = _calculate_nsii_from_sii(
            sii_values.values,
            sii_values.n_players,
            sii_values.max_order,
            sii_values.interaction_lookup,
        )
        return InteractionValues(
            values=n_sii_values,
            index="nSII",
            max_order=sii_values.max_order,
            min_order=sii_values.min_order,
            n_players=sii_values.n_players,
            interaction_lookup=sii_values.interaction_lookup,
            estimated=sii_values.estimated,
            estimation_budget=sii_values.estimation_budget,
        )
    elif approximator is not None:
        return _calculate_nsii_from_sii(
            sii_values, approximator.n, approximator.max_order, approximator.interaction_lookup
        )
    elif n is not None and max_order is not None:
        if interaction_lookup is None:
            interaction_lookup = _generate_interaction_lookup(n, 1, max_order)
        return _calculate_nsii_from_sii(sii_values, n, max_order, interaction_lookup)
    else:
        raise ValueError(
            "If the SII values are not provided as InteractionValues, the approximator "
            "or the number of players and the maximum order of the approximation must be "
            "provided."
        )


def _calculate_nsii_from_sii(
    sii_values: np.ndarray[float],
    n: int,
    max_order: int,
    interaction_lookup: Optional[dict] = None,
) -> np.ndarray[float]:
    """Calculates the nSII values from the SII values.

    Args:
        sii_values: The SII values to transform.
        n: The number of players.
        max_order: The maximum order of the approximation.
        interaction_lookup: A dictionary that maps interactions to their index in the values
            vector. If `interaction_lookup` is not provided, it is computed from the `n_players`,
            `min_order`, and `max_order` parameters. Defaults to `None`.

    Returns:
        The nSII values.
    """
    # compute nSII values from SII values
    bernoulli_numbers = bernoulli(max_order)
    nsii_values = np.zeros_like(sii_values)
    # all subsets S with 1 <= |S| <= max_order
    for subset in powerset(set(range(n)), min_size=1, max_size=max_order):
        interaction_index = interaction_lookup[subset]
        interaction_size = len(subset)
        n_sii_value = sii_values[interaction_index]
        # go over all subsets T of length |S| + 1, ..., n that contain S
        for T in powerset(set(range(n)), min_size=interaction_size + 1, max_size=max_order):
            if set(subset).issubset(T):
                effect_index = interaction_lookup[T]  # get the index of T
                effect_value = sii_values[effect_index]  # get the effect of T
                bernoulli_factor = bernoulli_numbers[len(T) - interaction_size]
                n_sii_value += bernoulli_factor * effect_value
        nsii_values[interaction_index] = n_sii_value
    return nsii_values


def convert_nsii_into_one_dimension(
    n_sii_values: InteractionValues,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Converts the nSII values into one-dimensional values.

    Args:
        n_sii_values: The nSII values to convert.

    Returns:
        The positive and negative one-dimensional values.
    """
    if n_sii_values.index != "nSII":
        raise ValueError(
            "Only nSII values can be converted into one-dimensional nSII values. Please use the "
            "transforms_sii_to_nsii method to convert SII values into nSII values."
        )
    max_order = n_sii_values.max_order
    min_order = n_sii_values.min_order
    n = n_sii_values.n_players

    pos_nsii_values = np.zeros(shape=(n,), dtype=float)
    neg_nsii_values = np.zeros(shape=(n,), dtype=float)

    for subset in powerset(set(range(n)), min_size=min_order, max_size=max_order):
        n_sii_value = n_sii_values[subset] / len(subset)  # distribute uniformly
        for player in subset:
            if n_sii_value >= 0:
                pos_nsii_values[player] += n_sii_value
            else:
                neg_nsii_values[player] += n_sii_value
    return pos_nsii_values, neg_nsii_values


def _generate_interaction_lookup(n: int, min_order: int, max_order: int) -> dict[tuple[int], int]:
    """Generates a lookup dictionary for interactions.

    Args:
        n: The number of players.
        min_order: The minimum order of the approximation.
        max_order: The maximum order of the approximation.

    Returns:
        A dictionary that maps interactions to their index in the values vector.
    """
    interaction_lookup = {
        interaction: i
        for i, interaction in enumerate(
            powerset(set(range(n)), min_size=min_order, max_size=max_order)
        )
    }
    return interaction_lookup
