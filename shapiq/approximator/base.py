"""This module contains the base approximator classes for the shapiq package."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
from scipy.special import binom

from shapiq.approximator.sampling import CoalitionSampler
from shapiq.game_theory.indices import get_computation_index
from shapiq.utils.sets import generate_interaction_lookup

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.games.base import Game
    from shapiq.interaction_values import InteractionValues

__all__ = [
    "Approximator",
]


ValidApproximationIndices = Literal[
    "SV", "BV", "SII", "BII", "k-SII", "STII", "FBII", "FSII", "kADD-SHAP", "CHII"
]


class Approximator(ABC):
    """This class is the base class for all approximators.

    Approximators are used to estimate the interaction values of a model or any value function.
    Different approximators can be used to estimate different interaction indices. Some can be used
    to estimate all indices.

    Attributes:
        n: The number of players.
        _grand_coalition_set: The set of players (starting from ``0`` to ``n - 1``).
        _grand_coalition_array: The array of players (starting from ``0`` to ``n``).
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated.
        top_order: If True, the approximation is performed only for the top order interactions. If
            False, the approximation is performed for all orders up to the specified order.
        min_order: The minimum order of the approximation. If top_order is ``True``, ``min_order``
            is equal to max_order. Otherwise, ``min_order`` is equal to ``0``.
        iteration_cost: The cost of a single iteration of the approximator.

    """

    valid_indices: tuple[ValidApproximationIndices] = tuple(get_args(ValidApproximationIndices))
    """The valid indices for the base approximator."""

    @abstractmethod
    def __init__(
        self,
        n: int,
        max_order: int,
        index: ValidApproximationIndices,
        *,
        top_order: bool,
        min_order: int = 0,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray[float] | None = None,
        random_state: int | None = None,
        initialize_dict: bool = True,
    ) -> None:
        """Initialize the Approximator.

        Args:
            n: The number of players.

            max_order: The maximum interaction order of the approximation.

            index: The interaction index to be estimated.

            top_order: If True, the approximation is performed only for the top order interactions.
                If False, the approximation is performed for all orders up to the specified order.

            min_order: The minimum interaction order of the approximation. Defaults to ``0``.

            pairing_trick: If True, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            random_state: The random state to use for the approximation. Defaults to ``None``. If
                not ``None``, the random state is used to seed the random number generator.

            initialize_dict: If True, initializes the interaction lookup dictionary. Defaults to
                ``True``. Note this is often ``True`` for estimators that estimate all interactions
                for each order. This is set to ``False`` for example in
                :class:`~shapiq.approximator.sparse.SPEX`.
        """
        # check if index is valid
        if index not in self.valid_indices:
            msg = f"Invalid index '{index}'. Valid indices are: {self.valid_indices}."
            raise ValueError(msg)

        # check if index can be approximated
        self.index: str = index
        self.approximation_index: str = get_computation_index(index)

        # get approximation parameters
        self.n: int = n
        self.top_order: bool = top_order
        self.max_order: int = max_order
        self.min_order: int = self.max_order if self.top_order else min_order
        self._grand_coalition_set = set(range(self.n))
        self._grand_coalition_tuple = tuple(range(self.n))
        self._grand_coalition_array: np.ndarray = np.arange(self.n + 1, dtype=int)
        self.iteration_cost: int = 1  # default value, can be overwritten by subclasses

        # The interaction_lookup is not initialized is some cases due to performance reasons
        if initialize_dict:
            self._interaction_lookup = generate_interaction_lookup(
                self.n,
                self.min_order,
                self.max_order,
            )
        else:
            self._interaction_lookup = {}

        # set up random state and random number generators
        self._random_state: int | None = random_state
        self._rng: np.random.Generator | None = np.random.default_rng(seed=self._random_state)

        # set up a coalition sampler
        self._big_M: int = 100_000_000  # large number for sampling weights
        if sampling_weights is None:  # init default sampling weights
            sampling_weights = self._init_sampling_weights()
        self._sampler = CoalitionSampler(
            n_players=self.n,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=self._random_state,
        )

    def __call__(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: Any,
    ) -> InteractionValues:
        """Calls the approximate method.

        This method is a wrapper around the `approximate` method. It is used to call the
        approximator with the given budget and game function.

        Args:
            budget: The budget for the approximation.

            game: The game function as a callable that takes a set of players and returns the value.

            **kwargs: Additional keyword arguments to pass to the `approximate` method.

        """
        return self.approximate(budget=budget, game=game, **kwargs)

    def set_random_state(self, random_state: int | None = None) -> None:
        """Sets the random state for the approximator.

        Args:
            random_state: The random state to set. If ``None``, no random state is set.

        """
        self._random_state = random_state
        self._rng = np.random.default_rng(seed=random_state)
        self._sampler.set_random_state(random_state)

    @abstractmethod
    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> InteractionValues:
        """Approximates the interaction values.

        Abstract method that needs to be implemented for each approximator.

        Args:
            budget: The budget for the approximation.

            game: The game function.

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.

        Returns:
            The interaction values.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        msg = "The approximate method must be implemented in the subclass."
        raise NotImplementedError(msg)  # pragma: no cover

    def _init_sampling_weights(self) -> np.ndarray:
        """Initializes the weights for sampling subsets.

        The sampling weights are of size ``n + 1`` and indexed by the size of the subset. The edges
        All weights are set to ``_big_M``, if ``size < order`` or ``size > n - order`` to ensure efficiency.

        Returns:
            The weights for sampling subsets of size ``s`` in shape ``(n + 1,)``.

        """
        weight_vector = np.zeros(shape=self.n + 1)
        if self.index in ["FBII"]:
            try:
                for coalition_size in range(self.n + 1):
                    weight_vector[coalition_size] = binom(self.n, coalition_size) / 2**self.n
            except OverflowError:
                for coalition_size in range(self.n + 1):
                    weight_vector[coalition_size] = (
                        1
                        / np.sqrt(2 * np.pi * 0.5)
                        * np.exp(-(coalition_size - self.n / 2) * +2 / (self.n / 2))
                    )
                warnings.warn(
                    "The weights are approximated for n > 1000. While this is very close to the truth for sets for size in the region n/2, the approximation is not exact for size n or 0.",
                    stacklevel=2,
                )
        else:
            for coalition_size in range(self.n + 1):
                if (coalition_size < self.max_order) or (coalition_size > self.n - self.max_order):
                    # prioritize these subsets
                    weight_vector[coalition_size] = self._big_M
                else:
                    # KernelSHAP sampling weights
                    weight_vector[coalition_size] = 1 / (coalition_size * (self.n - coalition_size))
        return weight_vector / np.sum(weight_vector)

    def _init_result(self, dtype: np.dtype | float = float) -> np.ndarray:
        """Initializes the result array for the approximation.

        Initializes the result array. The result array is a 1D array of size n_interactions as
        determined by the interaction_lookup dictionary.

        Args:
            dtype: The data type of the result array. Defaults to float.

        Returns:
            The result array.

        """
        return np.zeros(len(self._interaction_lookup), dtype=dtype)

    @property
    def _order_iterator(self) -> range:
        """Returns an iterator over the orders of the approximation.

        Returns:
            The iterator.

        """
        return range(self.min_order, self.max_order + 1)

    @staticmethod
    def _calc_iteration_count(budget: int, batch_size: int, iteration_cost: int) -> tuple[int, int]:
        """Calculate the number of iterations and the size of the last batch.

        Computes the number of iterations and the size of the last batch given the batch size and
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
            msg = "Cannot compare Approximator with other types."
            raise TypeError(msg)
        return not (
            self.n != other.n
            or self.max_order != other.max_order
            or self.index != other.index
            or self.top_order != other.top_order
            or self._random_state != other._random_state
        )

    def __ne__(self, other: object) -> bool:
        """Checks if two Approximator objects are not equal."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Returns the hash of the Approximator object."""
        return hash((self.n, self.max_order, self.index, self.top_order, self._random_state))

    @property
    def approximator_id(self) -> int:
        """Returns the ID of the approximator."""
        return hash(self)

    @property
    def interaction_lookup(self) -> dict[tuple[int, ...], int]:
        """Returns the interaction lookup dictionary."""
        return self._interaction_lookup

    @staticmethod
    def aggregate_interaction_values(
        base_interactions: InteractionValues,
        order: int | None = None,
    ) -> InteractionValues:
        """Aggregates the interaction values.

        Args:
            base_interactions: The base interaction values to aggregate.
            order: The order of the aggregation. For example, the order of the k-SII aggregation.
                If ``None`` (default), the maximum order of the base interactions is used.

        Returns:
            The aggregated interaction values.

        """
        from shapiq.game_theory.aggregation import aggregate_base_interaction

        return aggregate_base_interaction(base_interactions, order=order)
