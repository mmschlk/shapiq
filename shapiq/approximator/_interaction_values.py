import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from approximator._config import AVAILABLE_INDICES
from utils.sets import generate_interaction_lookup

from shapiq.utils import powerset


@dataclass
class InteractionValues:
    """This class contains the interaction values as estimated by an approximator.

    Attributes:
        values: The interaction values of the model in vectorized form.
        index: The interaction index estimated. Available indices are 'SII', 'kSII', 'STI', and
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
            self.interaction_lookup = generate_interaction_lookup(
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
