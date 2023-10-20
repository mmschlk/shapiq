"""This module contains the base approximator classes for the shapiq package."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np


AVAILABLE_INDICES = {"SII", "nSII", "STI", "FSI"}


@dataclass
class InteractionValues:
    """ This class contains the interaction values as estimated by an approximator.

    Attributes:
        values: The interaction values of the model. Mapping from order to the interaction values.
        index: The interaction index estimated. Available indices are 'SII', 'nSII', 'STI', and
            'FSI'.
        order: The order of the approximation.
    """
    values: dict[int, np.ndarray]
    index: str
    order: int

    def __post_init__(self) -> None:
        """Checks if the index is valid."""
        if self.index not in ["SII", "nSII", "STI", "FSI"]:
            raise ValueError(
                f"Index {self.index} is not valid. "
                f"Available indices are 'SII', 'nSII', 'STI', and 'FSI'."
            )
        if self.order < 1 or self.order != max(self.values.keys()):
            raise ValueError(
                f"Order {self.order} is not valid. "
                f"Order should be a positive integer equal to the maximum key of the values."
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

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated.
        top_order: If True, the approximation is performed only for the top order interactions. If
            False, the approximation is performed for all orders up to the specified order.
        min_order: The minimum order of the approximation. If top_order is True, min_order is equal
            to max_order. Otherwise, min_order is equal to 1.
    """

    def __init__(self, n: int, max_order: int, index: str, top_order: bool) -> None:
        """Initializes the approximator."""
        self.index: str = index
        if self.index not in AVAILABLE_INDICES:
            raise ValueError(
                f"Index {self.index} is not valid. "
                f"Available indices are {AVAILABLE_INDICES}."
            )
        self.n: int = n
        self.N: set = set(range(self.n))
        self.max_order: int = max_order
        self.top_order: bool = top_order
        self.min_order: int = self.max_order if self.top_order else 1

    @abstractmethod
    def approximate(
            self,
            budget: int,
            game: Callable[[Union[set, tuple]], float],
            *args, **kwargs
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
        raise NotImplementedError

    def _init_result(self, dtype=float) -> dict[int, np.ndarray]:
        """Initializes the result dictionary.

        Args:
            dtype: The data type of the result dictionary values. Defaults to float.

        Returns:
            The result dictionary.
        """
        result = {s: np.zeros(self.n + 1, dtype=dtype)
                  for s in range(self.min_order, self.max_order + 1)}
        return result

    def _finalize_result(self, result) -> InteractionValues:
        """Finalizes the result dictionary.

        Args:
            result: The result dictionary.

        Returns:
            The interaction values.
        """
        return InteractionValues(result, self.index, self.max_order)
