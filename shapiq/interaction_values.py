"""This module contains the InteractionValues Dataclass, which is used to store the interaction
scores."""
import copy
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from utils import generate_interaction_lookup, powerset

AVAILABLE_INDICES = {"k-SII", "SII", "STI", "FSI", "SV", "BZF"}


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
            The interaction value. If the interaction is not present zero is returned.
        """
        item = tuple(sorted(item))
        try:
            return float(self.values[self.interaction_lookup[item]])
        except KeyError:
            return 0.0

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
            or self.min_order != other.min_order
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
        return hash(
            (
                self.index,
                self.max_order,
                self.min_order,
                self.n_players,
                tuple(self.values.flatten()),
            )
        )

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

    def __add__(self, other: Union["InteractionValues", int, float]) -> "InteractionValues":
        """Adds two InteractionValues objects together or a scalar."""
        n_players, min_order, max_order = self.n_players, self.min_order, self.max_order
        if isinstance(other, InteractionValues):
            if self.index != other.index:  # different indices
                raise ValueError(
                    f"Cannot add InteractionValues with different indices {self.index} and "
                    f"{other.index}."
                )
            if (
                self.interaction_lookup != other.interaction_lookup
                or self.n_players != other.n_players
                or self.min_order != other.min_order
                or self.max_order != other.max_order
            ):  # different interactions
                warnings.warn(
                    "Adding InteractionValues with different interactions. Interactions will be "
                    "merged and added together. The resulting InteractionValues will have the "
                    "union of the interactions of the two original InteractionValues."
                )
                interaction_lookup = {**self.interaction_lookup}
                position = len(self.interaction_lookup)
                values_to_add = []
                added_values = self.values.copy()
                for interaction in other.interaction_lookup:
                    if interaction not in interaction_lookup:
                        interaction_lookup[interaction] = position
                        position += 1
                        values_to_add.append(other[interaction])
                    else:
                        added_values[interaction_lookup[interaction]] += other[interaction]
                added_values = np.concatenate((added_values, np.asarray(values_to_add)))
                # adjust n_players, min_order, and max_order
                n_players = max(self.n_players, other.n_players)
                min_order = min(self.min_order, other.min_order)
                max_order = max(self.max_order, other.max_order)
            else:  # basic case with same interactions
                added_values = self.values + other.values
                interaction_lookup = self.interaction_lookup
        elif isinstance(other, (int, float)):
            added_values = self.values.copy() + other
            interaction_lookup = self.interaction_lookup.copy()
        else:
            raise TypeError("Cannot add InteractionValues with object of type " f"{type(other)}.")
        return InteractionValues(
            values=added_values,
            index=self.index,
            max_order=max_order,
            n_players=n_players,
            min_order=min_order,
            interaction_lookup=interaction_lookup,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
        )

    def __radd__(self, other: Union["InteractionValues", int, float]) -> "InteractionValues":
        """Adds two InteractionValues objects together or a scalar."""
        return self.__add__(other)

    def __neg__(self):
        """Negates the InteractionValues object."""
        return InteractionValues(
            values=-self.values,
            index=self.index,
            max_order=self.max_order,
            n_players=self.n_players,
            min_order=self.min_order,
            interaction_lookup=self.interaction_lookup,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
        )

    def __sub__(self, other: Union["InteractionValues", int, float]) -> "InteractionValues":
        """Subtracts two InteractionValues objects or a scalar."""
        return self.__add__(-other)

    def __rsub__(self, other: Union["InteractionValues", int, float]) -> "InteractionValues":
        """Subtracts two InteractionValues objects or a scalar."""
        return (-self).__add__(other)

    def __mul__(self, other: Union[int, float]) -> "InteractionValues":
        """Multiplies an InteractionValues object by a scalar."""
        return InteractionValues(
            values=self.values * other,
            index=self.index,
            max_order=self.max_order,
            n_players=self.n_players,
            min_order=self.min_order,
            interaction_lookup=self.interaction_lookup,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
        )

    def __rmul__(self, other: Union[int, float]) -> "InteractionValues":
        """Multiplies an InteractionValues object by a scalar."""
        return self.__mul__(other)
