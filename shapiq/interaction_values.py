"""InteractionValues data-class, which is used to store the interaction scores."""

from __future__ import annotations

import contextlib
import copy
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from .game_theory.indices import (
    ALL_AVAILABLE_INDICES,
    index_generalizes_bv,
    index_generalizes_sv,
    is_empty_value_the_baseline,
    is_index_aggregated,
)
from .utils.errors import raise_deprecation_warning
from .utils.sets import generate_interaction_lookup

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from shapiq.typing import JSONType


SAVE_JSON_DEPRECATION_MSG = (
    "Saving InteractionValues not as a JSON file is deprecated. "
    "The parameters `as_pickle` and `as_npz` will be removed in the future. "
)


@dataclass
class InteractionValues:
    """This class contains the interaction values as estimated by an approximator.

    Attributes:
        values: The interaction values of the model in vectorized form.
        index: The interaction index estimated. All available indices are defined in
            ``ALL_AVAILABLE_INDICES``.
        max_order: The order of the approximation.
        n_players: The number of players.
        min_order: The minimum order of the approximation. Defaults to ``0``.
        interaction_lookup: A dictionary that maps interactions to their index in the values
            vector. If ``interaction_lookup`` is not provided, it is computed from the ``n_players``,
            ``min_order``, and `max_order` parameters. Defaults to ``None``.
        estimated: Whether the interaction values are estimated or not. Defaults to ``True``.
        estimation_budget: The budget used for the estimation. Defaults to ``None``.
        baseline_value: The value of the baseline interaction also known as 'empty prediction' or
            ``'empty value'`` since it denotes the value of the empty coalition (empty set). If not
            provided it is searched for in the values vector (raising an Error if not found).
            Defaults to ``None``.

    Raises:
        UserWarning: If the index is not a valid index as defined in ``ALL_AVAILABLE_INDICES``.
        TypeError: If the baseline value is not a number.

    """

    values: np.ndarray
    index: str
    max_order: int
    n_players: int
    min_order: int
    baseline_value: float
    interaction_lookup: dict[tuple[int, ...], int] = None  # type: ignore[assignment]
    estimated: bool = True
    estimation_budget: int = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Checks if the index is valid."""
        if self.index not in ALL_AVAILABLE_INDICES:
            warn(
                UserWarning(
                    f"Index {self.index} is not a valid index as defined in "
                    f"{ALL_AVAILABLE_INDICES}. This might lead to unexpected behavior.",
                ),
                stacklevel=2,
            )

        # set BV or SV if max_order is 1
        if self.max_order == 1:
            if index_generalizes_bv(self.index):
                self.index = "BV"
            if index_generalizes_sv(self.index):
                self.index = "SV"

        # populate interaction_lookup and reverse_interaction_lookup
        if self.interaction_lookup is None:
            self.interaction_lookup = generate_interaction_lookup(
                self.n_players,
                self.min_order,
                self.max_order,
            )

        if not isinstance(self.baseline_value, int | float):
            msg = f"Baseline value must be provided as a number. Got {self.baseline_value}."
            raise TypeError(msg)

        # check if () is in the interaction_lookup if min_order is 0 -> add it to the end
        if self.min_order == 0 and () not in self.interaction_lookup:
            self.interaction_lookup[()] = len(self.interaction_lookup)
            self.values = np.concatenate((self.values, np.array([self.baseline_value])))

    @property
    def dict_values(self) -> dict[tuple[int, ...], float]:
        """Getter for the dict directly mapping from all interactions to scores."""
        return {
            interaction: float(self.values[self.interaction_lookup[interaction]])
            for interaction in self.interaction_lookup
        }

    def to_json_file(
        self,
        path: Path,
        *,
        desc: str | None = None,
        created_from: object | None = None,
        **kwargs: JSONType,
    ) -> None:
        """Saves the InteractionValues object to a JSON file.

        Args:
            path: The path to the JSON file.
            desc: A description of the InteractionValues object. Defaults to ``None``.
            created_from: An object from which the InteractionValues object was created. Defaults to
                ``None``.
            **kwargs: Additional parameters to store in the metadata of the JSON file.
        """
        from shapiq.utils.saving import interactions_to_dict, make_file_metadata, save_json

        file_metadata = make_file_metadata(
            object_to_store=self,
            data_type="interaction_values",
            desc=desc,
            created_from=created_from,
            parameters=kwargs,
        )
        json_data = {
            **file_metadata,
            "metadata": {
                "n_players": self.n_players,
                "index": self.index,
                "max_order": self.max_order,
                "min_order": self.min_order,
                "estimated": self.estimated,
                "estimation_budget": self.estimation_budget,
                "baseline_value": self.baseline_value,
            },
            "data": interactions_to_dict(interactions=self.dict_values),
        }
        save_json(json_data, path)

    @classmethod
    def from_json_file(cls, path: Path) -> InteractionValues:
        """Loads an InteractionValues object from a JSON file.

        Args:
            path: The path to the JSON file. Note that the path must end with `'.json'`.

        Returns:
            The InteractionValues object loaded from the JSON file.

        Raises:
            ValueError: If the path does not end with `'.json'`.
        """
        from shapiq.utils.saving import dict_to_lookup_and_values

        if not path.name.endswith(".json"):
            msg = f"Path {path} does not end with .json. Cannot load InteractionValues."
            raise ValueError(msg)

        with path.open("r", encoding="utf-8") as file:
            json_data = json.load(file)

        metadata = json_data["metadata"]
        interaction_dict = json_data["data"]
        interaction_lookup, values = dict_to_lookup_and_values(interaction_dict)

        return cls(
            values=values,
            index=metadata["index"],
            max_order=metadata["max_order"],
            n_players=metadata["n_players"],
            min_order=metadata["min_order"],
            interaction_lookup=interaction_lookup,
            estimated=metadata["estimated"],
            estimation_budget=metadata["estimation_budget"],
            baseline_value=metadata["baseline_value"],
        )

    def sparsify(self, threshold: float = 1e-3) -> None:
        """Manually sets values close to zero actually to zero (removing values).

        Args:
            threshold: The threshold value below which interactions are zeroed out. Defaults to
                1e-3.

        """
        # find interactions to remove in self.values
        interactions_to_remove: set[int] = set(np.where(np.abs(self.values) < threshold)[0])
        new_values = np.delete(self.values, list(interactions_to_remove))
        new_interaction_lookup = {}
        for index, _interaction in enumerate(self.interaction_lookup):
            if index not in interactions_to_remove:
                interaction = tuple(sorted(_interaction))
                new_interaction_lookup[interaction] = len(new_interaction_lookup)
        self.values = new_values
        self.interaction_lookup = new_interaction_lookup

    def get_top_k_interactions(self, k: int) -> InteractionValues:
        """Returns the top k interactions.

        Args:
            k: The number of top interactions to return.

        Returns:
            The top k interactions as an InteractionValues object.

        """
        top_k_indices = np.argsort(np.abs(self.values))[::-1][:k]
        new_values = np.zeros(k, dtype=float)
        new_interaction_lookup = {}
        for interaction_pos, interaction in enumerate(self.interaction_lookup):
            if interaction_pos in top_k_indices:
                new_position = len(new_interaction_lookup)
                new_values[new_position] = float(self[interaction_pos])
                new_interaction_lookup[interaction] = new_position
        return InteractionValues(
            values=new_values,
            index=self.index,
            max_order=self.max_order,
            n_players=self.n_players,
            min_order=self.min_order,
            interaction_lookup=new_interaction_lookup,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
            baseline_value=self.baseline_value,
        )

    def get_top_k(
        self, k: int, *, as_interaction_values: bool = True
    ) -> InteractionValues | tuple[dict, list[tuple]]:
        """Returns the top k interactions.

        Args:
            k: The number of top interactions to return.
            as_interaction_values: Whether to return the top `k` interactions as an InteractionValues
                object. Defaults to ``False``.

        Returns:
            The top k interactions as a dictionary and a sorted list of tuples.

        Examples:
            >>> interaction_values = InteractionValues(
            ...     values=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            ...     interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
            ...     index="SII",
            ...     max_order=2,
            ...     n_players=3,
            ...     min_order=1,
            ...     baseline_value=0.0,
            ... )
            >>> top_k_interactions, sorted_top_k_interactions = interaction_values.get_top_k(2, False)
            >>> top_k_interactions
            {(0, 2): 0.5, (1, 0): 0.6}
            >>> sorted_top_k_interactions
            [((1, 0), 0.6), ((0, 2), 0.5)]

        """
        if as_interaction_values:
            return self.get_top_k_interactions(k)
        top_k_indices = np.argsort(np.abs(self.values))[::-1][:k]
        top_k_interactions = {}
        for interaction, index in self.interaction_lookup.items():
            if index in top_k_indices:
                top_k_interactions[interaction] = self.values[index]
        sorted_top_k_interactions = [
            (interaction, top_k_interactions[interaction])
            for interaction in sorted(top_k_interactions, key=top_k_interactions.get, reverse=True)
        ]
        return top_k_interactions, sorted_top_k_interactions

    def __repr__(self) -> str:
        """Returns the representation of the InteractionValues object."""
        representation = "InteractionValues(\n"
        representation += (
            f"    index={self.index}, max_order={self.max_order}, min_order={self.min_order}"
            f", estimated={self.estimated}, estimation_budget={self.estimation_budget},\n"
            f"    n_players={self.n_players}, baseline_value={self.baseline_value}\n)"
        )
        return representation

    def __str__(self) -> str:
        """Returns the string representation of the InteractionValues object."""
        representation = self.__repr__()
        representation = representation[:-2]  # remove the last "\n)" and add values
        _, sorted_top_10_interactions = self.get_top_k(
            10, as_interaction_values=False
        )  # get top 10 interactions
        # add values to string representation
        representation += ",\n    Top 10 interactions:\n"
        for interaction, value in sorted_top_10_interactions:
            representation += f"        {interaction}: {value}\n"
        representation += ")"
        return representation

    def __len__(self) -> int:
        """Returns the length of the InteractionValues object."""
        return len(self.values)  # might better to return the theoretical no. of interactions

    def __iter__(self) -> np.nditer:
        """Returns an iterator over the values of the InteractionValues object."""
        return np.nditer(self.values)

    def __getitem__(self, item: int | tuple[int, ...]) -> float:
        """Returns the score for the given interaction.

        Args:
            item: The interaction as a tuple of integers for which to return the score. If ``item`` is
                an integer it serves as the index to the values vector.

        Returns:
            The interaction value. If the interaction is not present zero is returned.

        """
        if isinstance(item, int):
            return float(self.values[item])
        item = tuple(sorted(item))
        try:
            return float(self.values[self.interaction_lookup[item]])
        except KeyError:
            return 0.0

    def __setitem__(self, item: int | tuple[int, ...], value: float) -> None:
        """Sets the score for the given interaction.

        Args:
            item: The interaction as a tuple of integers for which to set the score. If ``item`` is an
                integer it serves as the index to the values vector.
            value: The value to set for the interaction.

        Raises:
            KeyError: If the interaction is not found in the InteractionValues object.

        """
        try:
            if isinstance(item, int):
                self.values[item] = value
            else:
                item = tuple(sorted(item))
                self.values[self.interaction_lookup[item]] = value
        except Exception as e:
            msg = f"Interaction {item} not found in the InteractionValues. Unable to set a value."
            raise KeyError(msg) from e

    def __eq__(self, other: object) -> bool:
        """Checks if two InteractionValues objects are equal.

        Args:
            other: The other InteractionValues object.

        Returns:
            True if the two objects are equal, False otherwise.

        """
        if not isinstance(other, InteractionValues):
            msg = "Cannot compare InteractionValues with other types."
            raise TypeError(msg)
        if (
            self.index != other.index
            or self.max_order != other.max_order
            or self.min_order != other.min_order
            or self.n_players != other.n_players
            or self.baseline_value != other.baseline_value
        ):
            return False
        if not np.allclose(self.values, other.values):
            return False
        return self.interaction_lookup == other.interaction_lookup

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
            ),
        )

    def __copy__(self) -> InteractionValues:
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
            baseline_value=self.baseline_value,
        )

    def __add__(self, other: InteractionValues | float) -> InteractionValues:
        """Adds two InteractionValues objects together or a scalar."""
        n_players, min_order, max_order = self.n_players, self.min_order, self.max_order
        if isinstance(other, InteractionValues):
            if self.index != other.index:  # different indices
                msg = (
                    f"Cannot add InteractionValues with different indices {self.index} and "
                    f"{other.index}."
                )
                raise ValueError(msg)
            if (
                self.interaction_lookup != other.interaction_lookup
                or self.n_players != other.n_players
                or self.min_order != other.min_order
                or self.max_order != other.max_order
            ):  # different interactions but addable
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
                baseline_value = self.baseline_value + other.baseline_value
            else:  # basic case with same interactions
                added_values = self.values + other.values
                interaction_lookup = self.interaction_lookup
                baseline_value = self.baseline_value + other.baseline_value
        elif isinstance(other, int | float):
            added_values = self.values.copy() + other
            interaction_lookup = self.interaction_lookup.copy()
            baseline_value = self.baseline_value + other
        else:
            msg = f"Cannot add InteractionValues with object of type {type(other)}."
            raise TypeError(msg)
        return InteractionValues(
            values=added_values,
            index=self.index,
            max_order=max_order,
            n_players=n_players,
            min_order=min_order,
            interaction_lookup=interaction_lookup,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
            baseline_value=baseline_value,
        )

    def __radd__(self, other: InteractionValues | float) -> InteractionValues:
        """Adds two InteractionValues objects together or a scalar."""
        return self.__add__(other)

    def __neg__(self) -> InteractionValues:
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
            baseline_value=-self.baseline_value,
        )

    def __sub__(self, other: InteractionValues | float) -> InteractionValues:
        """Subtracts two InteractionValues objects or a scalar."""
        return self.__add__(-other)

    def __rsub__(self, other: InteractionValues | float) -> InteractionValues:
        """Subtracts two InteractionValues objects or a scalar."""
        return (-self).__add__(other)

    def __mul__(self, other: float) -> InteractionValues:
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
            baseline_value=self.baseline_value * other,
        )

    def __rmul__(self, other: float) -> InteractionValues:
        """Multiplies an InteractionValues object by a scalar."""
        return self.__mul__(other)

    def __abs__(self) -> InteractionValues:
        """Returns the absolute values of the InteractionValues object."""
        return InteractionValues(
            values=np.abs(self.values),
            index=self.index,
            max_order=self.max_order,
            n_players=self.n_players,
            min_order=self.min_order,
            interaction_lookup=self.interaction_lookup,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
            baseline_value=self.baseline_value,
        )

    def get_n_order_values(self, order: int) -> np.ndarray:
        """Returns the interaction values of a specific order as a numpy array.

        Note:
            Depending on the order and number of players the resulting array might be sparse and
            very large.

        Args:
            order: The order of the interactions to return.

        Returns:
            The interaction values of the specified order as a numpy array of shape ``(n_players,)``
            for order ``1`` and ``(n_players, n_players)`` for order ``2``, etc.

        Raises:
            ValueError: If the order is less than ``1``.

        """
        from itertools import permutations

        if order < 1:
            msg = "Order must be greater or equal to 1."
            raise ValueError(msg)
        values_shape = tuple([self.n_players] * order)
        values = np.zeros(values_shape, dtype=float)
        for interaction in self.interaction_lookup:
            if len(interaction) != order:
                continue
            # get all orderings of the interaction (e.g. (0, 1) and (1, 0) for interaction (0, 1))
            for perm in permutations(interaction):
                values[perm] = self[interaction]

        return values

    def get_n_order(
        self,
        order: int | None = None,
        min_order: int | None = None,
        max_order: int | None = None,
    ) -> InteractionValues:
        """Select particular order of interactions.

        Creates a new InteractionValues object containing only the interactions within the
        specified order range.

        You can specify:
            - `order`: to select interactions of a single specific order (e.g., all pairwise
                interactions).
            - `min_order` and/or `max_order`: to select a range of interaction orders.
            - If `order` and `min_order`/`max_order` are both set, `min_order` and `max_order` will
                override the `order` value.

        Example:
            >>> interaction_values = InteractionValues(
            ...     values=np.array([1, 2, 3, 4, 5, 6, 7]),
            ...     interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5, (0, 1, 2): 6},
            ...     index="SII",
            ...     max_order=3,
            ...     n_players=3,
            ...     min_order=1,
            ...     baseline_value=0.0,
            ... )
            >>> interaction_values.get_n_order(order=1).dict_values
            {(0,): 1.0, (1,): 2.0, (2,): 3.0}
            >>> interaction_values.get_n_order(min_order=1, max_order=2).dict_values
            {(0,): 1.0, (1,): 2.0, (2,): 3.0, (0, 1): 4.0, (0, 2): 5.0, (1, 2): 6.0}
            >>> interaction_values.get_n_order(min_order=2).dict_values
            {(0, 1): 4.0, (0, 2): 5.0, (1, 2): 6.0, (0, 1, 2): 7.0}

        Args:
            order: The order of the interactions to return. Defaults to ``None`` which requires
                ``min_order`` or ``max_order`` to be set.
            min_order: The minimum order of the interactions to return. Defaults to ``None`` which
                sets it to the order.
            max_order: The maximum order of the interactions to return. Defaults to ``None`` which
                sets it to the order.

        Returns:
            The interaction values of the specified order.

        Raises:
            ValueError: If all three parameters are set to ``None``.
        """
        if order is None and min_order is None and max_order is None:
            msg = "Either order, min_order or max_order must be set."
            raise ValueError(msg)

        if order is not None:
            max_order = order if max_order is None else max_order
            min_order = order if min_order is None else min_order
        else:  # order is None
            min_order = self.min_order if min_order is None else min_order
            max_order = self.max_order if max_order is None else max_order

        if min_order > max_order:
            msg = f"min_order ({min_order}) must be less than or equal to max_order ({max_order})."
            raise ValueError(msg)

        new_values = []
        new_interaction_lookup = {}
        for interaction in self.interaction_lookup:
            if len(interaction) < min_order or len(interaction) > max_order:
                continue
            interaction_idx = len(new_interaction_lookup)
            new_values.append(self[interaction])
            new_interaction_lookup[interaction] = interaction_idx

        return InteractionValues(
            values=np.array(new_values),
            index=self.index,
            max_order=max_order,
            n_players=self.n_players,
            min_order=min_order,
            interaction_lookup=new_interaction_lookup,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
            baseline_value=self.baseline_value,
        )

    def get_subset(self, players: list[int]) -> InteractionValues:
        """Selects a subset of players from the InteractionValues object.

        Args:
            players: List of players to select from the InteractionValues object.

        Returns:
            InteractionValues: Filtered InteractionValues object containing only values related to
            selected players.

        Example:
            >>> interaction_values = InteractionValues(
            ...     values=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            ...     interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
            ...     index="SII",
            ...     max_order=2,
            ...     n_players=3,
            ...     min_order=1,
            ...     baseline_value=0.0,
            ... )
            >>> interaction_values.get_subset([0, 1]).dict_values
            {(0,): 0.1, (1,): 0.2, (0, 1): 0.3}
            >>> interaction_values.get_subset([0, 2]).dict_values
            {(0,): 0.1, (2,): 0.3, (0, 2): 0.4}
            >>> interaction_values.get_subset([1]).dict_values
            {(1,): 0.2}

        """
        keys = self.interaction_lookup.keys()
        idx, keys_in_subset = [], []
        for i, key in enumerate(keys):
            if all(p in players for p in key):
                idx.append(i)
                keys_in_subset.append(key)
        new_values = self.values[idx]
        new_interaction_lookup = {key: index for index, key in enumerate(keys_in_subset)}
        n_players = self.n_players - len(players)
        return InteractionValues(
            values=new_values,
            index=self.index,
            max_order=self.max_order,
            n_players=n_players,
            min_order=self.min_order,
            interaction_lookup=new_interaction_lookup,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
            baseline_value=self.baseline_value,
        )

    def save(self, path: Path, *, as_pickle: bool = False, as_npz: bool = False) -> None:
        """Save the InteractionValues object to a file.

        By default, the InteractionValues object is saved as a JSON file.

        Args:
            path: The path to save the InteractionValues object to.
            as_pickle: Whether to save the InteractionValues object as a pickle file (``True``).
            as_npz: Whether to save the InteractionValues object as a ``npz`` file (``True``).

        Raises:
            DeprecationWarning: If `as_pickle` or `as_npz` is set to ``True``, a deprecation
                warning is raised
        """
        # check if the directory exists
        directory = Path(path).parent
        if not Path(directory).exists():
            with contextlib.suppress(FileNotFoundError):
                Path(directory).mkdir(parents=True, exist_ok=True)
        if as_pickle:
            raise_deprecation_warning(
                message=SAVE_JSON_DEPRECATION_MSG, deprecated_in="1.3.1", removed_in="1.4.0"
            )
            with Path(path).open("wb") as file:
                pickle.dump(self, file)
        elif as_npz:
            raise_deprecation_warning(
                message=SAVE_JSON_DEPRECATION_MSG, deprecated_in="1.3.1", removed_in="1.4.0"
            )
            # save object as npz file
            np.savez(
                path,
                values=self.values,
                index=self.index,
                max_order=self.max_order,
                n_players=self.n_players,
                min_order=self.min_order,
                interaction_lookup=self.interaction_lookup,
                estimated=self.estimated,
                estimation_budget=self.estimation_budget,
                baseline_value=self.baseline_value,
            )
        else:
            self.to_json_file(path)

    @classmethod
    def load(cls, path: Path | str) -> InteractionValues:
        """Load an InteractionValues object from a file.

        Args:
            path: The path to load the InteractionValues object from.

        Returns:
            The loaded InteractionValues object.

        """
        path = Path(path)
        # check if path ends with .json
        if path.name.endswith(".json"):
            return cls.from_json_file(path)

        raise_deprecation_warning(
            SAVE_JSON_DEPRECATION_MSG, deprecated_in="1.3.1", removed_in="1.4.0"
        )

        # try loading as npz file
        try:
            data = np.load(path, allow_pickle=True)
            return InteractionValues(
                values=data["values"],
                index=str(data["index"]),
                max_order=int(data["max_order"]),
                n_players=int(data["n_players"]),
                min_order=int(data["min_order"]),
                interaction_lookup=data["interaction_lookup"].item(),
                estimated=bool(data["estimated"]),
                estimation_budget=data["estimation_budget"].item(),
                baseline_value=float(data["baseline_value"]),
            )
        except AttributeError:  # not a npz file
            pass
        with path.open("rb") as file:
            # TODO(mmshlk): https://github.com/mmschlk/shapiq/issues/413
            return pickle.load(file)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionValues:
        """Create an InteractionValues object from a dictionary.

        Args:
            data: The dictionary containing the data to create the InteractionValues object from.

        Returns:
            The InteractionValues object created from the dictionary.

        """
        return cls(
            values=data["values"],
            index=data["index"],
            max_order=data["max_order"],
            n_players=data["n_players"],
            min_order=data["min_order"],
            interaction_lookup=data["interaction_lookup"],
            estimated=data["estimated"],
            estimation_budget=data["estimation_budget"],
            baseline_value=data["baseline_value"],
        )

    def to_dict(self) -> dict:
        """Convert the InteractionValues object to a dictionary.

        Returns:
            The InteractionValues object as a dictionary.

        """
        return {
            "values": self.values,
            "index": self.index,
            "max_order": self.max_order,
            "n_players": self.n_players,
            "min_order": self.min_order,
            "interaction_lookup": self.interaction_lookup,
            "estimated": self.estimated,
            "estimation_budget": self.estimation_budget,
            "baseline_value": self.baseline_value,
        }

    def aggregate(
        self,
        others: Sequence[InteractionValues],
        aggregation: str = "mean",
    ) -> InteractionValues:
        """Aggregates InteractionValues objects using a specific aggregation method.

        Args:
            others: A list of InteractionValues objects to aggregate.
            aggregation: The aggregation method to use. Defaults to ``"mean"``. Other options are
                ``"median"``, ``"sum"``, ``"max"``, and ``"min"``.

        Returns:
            The aggregated InteractionValues object.

        Note:
            For documentation on the aggregation methods, see the ``aggregate_interaction_values()``
            function.

        """
        return aggregate_interaction_values([self, *others], aggregation)

    def plot_network(self, *, show: bool = True, **kwargs: Any) -> tuple[Figure, Axes] | None:
        """Visualize InteractionValues on a graph.

        Note:
            For arguments, see :func:`shapiq.plot.network.network_plot` and
                :func:`shapiq.plot.si_graph.si_graph_plot`.

        Args:
            show: Whether to show the plot. Defaults to ``True``.

            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            If show is ``False``, the function returns a tuple with the figure and the axis of the
                plot.
        """
        from shapiq.plot.network import network_plot

        if self.max_order > 1:
            return network_plot(
                interaction_values=self,
                show=show,
                **kwargs,
            )
        msg = (
            "InteractionValues contains only 1-order values,"
            "but requires also 2-order values for the network plot."
        )
        raise ValueError(msg)

    def plot_si_graph(self, *, show: bool = True, **kwargs: Any) -> tuple[Figure, Axes] | None:
        """Visualize InteractionValues as a SI graph.

        For arguments, see shapiq.plots.si_graph_plot().

        Returns:
            The SI graph as a tuple containing the figure and the axes.

        """
        from shapiq.plot.si_graph import si_graph_plot

        return si_graph_plot(self, show=show, **kwargs)

    def plot_stacked_bar(self, *, show: bool = True, **kwargs: Any) -> tuple[Figure, Axes] | None:
        """Visualize InteractionValues on a graph.

        For arguments, see shapiq.plots.stacked_bar_plot().

        Returns:
            The stacked bar plot as a tuple containing the figure and the axes.

        """
        from shapiq import stacked_bar_plot

        return stacked_bar_plot(self, show=show, **kwargs)

    def plot_force(
        self,
        feature_names: np.ndarray | None = None,
        *,
        show: bool = True,
        abbreviate: bool = True,
        contribution_threshold: float = 0.05,
    ) -> Figure | None:
        """Visualize InteractionValues on a force plot.

        For arguments, see shapiq.plots.force_plot().

        Args:
            feature_names: The feature names used for plotting. If no feature names are provided, the
                feature indices are used instead. Defaults to ``None``.
            show: Whether to show the plot. Defaults to ``False``.
            abbreviate: Whether to abbreviate the feature names or not. Defaults to ``True``.
            contribution_threshold: The threshold for contributions to be displayed in percent.
                Defaults to ``0.05``.

        Returns:
            The force plot as a matplotlib figure (if show is ``False``).

        """
        from .plot import force_plot

        return force_plot(
            self,
            feature_names=feature_names,
            show=show,
            abbreviate=abbreviate,
            contribution_threshold=contribution_threshold,
        )

    def plot_waterfall(
        self,
        feature_names: np.ndarray | None = None,
        *,
        show: bool = True,
        abbreviate: bool = True,
        max_display: int = 10,
    ) -> Axes | None:
        """Draws interaction values on a waterfall plot.

        Note:
            Requires the ``shap`` Python package to be installed.

        Args:
            feature_names: The feature names used for plotting. If no feature names are provided, the
                feature indices are used instead. Defaults to ``None``.
            show: Whether to show the plot. Defaults to ``False``.
            abbreviate: Whether to abbreviate the feature names or not. Defaults to ``True``.
            max_display: The maximum number of interactions to display. Defaults to ``10``.
        """
        from shapiq import waterfall_plot

        return waterfall_plot(
            self,
            feature_names=feature_names,
            show=show,
            max_display=max_display,
            abbreviate=abbreviate,
        )

    def plot_sentence(
        self,
        words: list[str],
        *,
        show: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, Axes] | None:
        """Plots the first order effects (attributions) of a sentence or paragraph.

        For arguments, see shapiq.plots.sentence_plot().

        Returns:
            If ``show`` is ``True``, the function returns ``None``. Otherwise, it returns a tuple
            with the figure and the axis of the plot.

        """
        from shapiq.plot.sentence import sentence_plot

        return sentence_plot(self, words, show=show, **kwargs)

    def plot_upset(self, *, show: bool = True, **kwargs: Any) -> Figure | None:
        """Plots the upset plot.

        For arguments, see shapiq.plot.upset_plot().

        Returns:
            The upset plot as a matplotlib figure (if show is ``False``).

        """
        from shapiq.plot.upset import upset_plot

        return upset_plot(self, show=show, **kwargs)


def aggregate_interaction_values(
    interaction_values: Sequence[InteractionValues],
    aggregation: str = "mean",
) -> InteractionValues:
    """Aggregates InteractionValues objects using a specific aggregation method.

    Args:
        interaction_values: A list of InteractionValues objects to aggregate.
        aggregation: The aggregation method to use. Defaults to ``"mean"``. Other options are
            ``"median"``, ``"sum"``, ``"max"``, and ``"min"``.

    Returns:
        The aggregated InteractionValues object.

    Example:
        >>> iv1 = InteractionValues(
        ...     values=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ...     interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
        ...     index="SII",
        ...     max_order=2,
        ...     n_players=3,
        ...     min_order=1,
        ...     baseline_value=0.0,
        ... )
        >>> iv2 = InteractionValues(
        ...     values=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),  # this iv is missing the (1, 2) value
        ...     interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4},  # no (1, 2)
        ...     index="SII",
        ...     max_order=2,
        ...     n_players=3,
        ...     min_order=1,
        ...     baseline_value=1.0,
        ... )
        >>> aggregate_interaction_values([iv1, iv2], "mean")
        InteractionValues(
            index=SII, max_order=2, min_order=1, estimated=True, estimation_budget=None,
            n_players=3, baseline_value=0.5,
            Top 10 interactions:
                (1, 2): 0.60
                (0, 2): 0.35
                (0, 1): 0.25
                (0,): 0.15
                (1,): 0.25
                (2,): 0.35
        )

    Note:
        The index of the aggregated InteractionValues object is set to the index of the first
        InteractionValues object in the list.

    Raises:
        ValueError: If the aggregation method is not supported.

    """

    def _aggregate(vals: list[float], method: str) -> float:
        """Does the actual aggregation of the values."""
        if method == "mean":
            return float(np.mean(vals))
        if method == "median":
            return float(np.median(vals))
        if method == "sum":
            return np.sum(vals)
        if method == "max":
            return np.max(vals)
        if method == "min":
            return np.min(vals)
        msg = f"Aggregation method {method} is not supported."
        raise ValueError(msg)

    # get all keys from all InteractionValues objects
    all_keys = set()
    for iv in interaction_values:
        all_keys.update(iv.interaction_lookup.keys())
    all_keys = sorted(all_keys)

    # aggregate the values
    new_values = np.zeros(len(all_keys), dtype=float)
    new_lookup = {}
    for i, key in enumerate(all_keys):
        new_lookup[key] = i
        values = [iv[key] for iv in interaction_values]
        new_values[i] = _aggregate(values, aggregation)

    max_order = max([iv.max_order for iv in interaction_values])
    min_order = min([iv.min_order for iv in interaction_values])
    n_players = max([iv.n_players for iv in interaction_values])
    baseline_value = _aggregate([iv.baseline_value for iv in interaction_values], aggregation)
    estimation_budget = interaction_values[0].estimation_budget

    return InteractionValues(
        values=new_values,
        index=interaction_values[0].index,
        max_order=max_order,
        n_players=n_players,
        min_order=min_order,
        interaction_lookup=new_lookup,
        estimated=True,
        estimation_budget=estimation_budget,
        baseline_value=baseline_value,
    )


def finalize_computed_interactions(
    interactions: InteractionValues,
    target_index: str | None = None,
) -> InteractionValues:
    """Finalizes computed InteractionValues to be interpretable.

    This function takes care of the following:
        - Aggregates the interactions if necessary. (e.g. from SII to k-SII)
        - Adjusts the baseline and empty value if necessary. (e.g. for Shapley indices the baseline
            value is the prediction of the model without any features - also called empty value, for
            Banzhaf the baseline value is not the empty prediction as Banzhaf does not fulfill the
            efficiency property)

    Args:
        interactions: The InteractionValues to finalize.
        target_index: The index to which the InteractionValues should be finalized. Defaults to
            ``None`` which means that the InteractionValues are finalized to the index of the
            InteractionValues object.

    Returns:
        The interaction values.

    Note:
        If you develop new approximators and computation methods, you should finalize the
        InteractionValues object before returning it to the user.

    Raises:
        ValueError: If the baseline value is not provided for SII and k-SII.

    """
    from .game_theory.aggregation import aggregate_base_interaction

    if target_index is None:
        target_index = interactions.index

    # aggregate the interactions if necessary
    if is_index_aggregated(target_index) and target_index != interactions.index:
        interactions = aggregate_base_interaction(interactions)

    # adjust the baseline and empty value if necessary
    if () in interactions.interaction_lookup:
        idx = interactions.interaction_lookup[()]
        empty_value = interactions[idx]
        if empty_value != interactions.baseline_value and interactions.index != "SII":
            if is_empty_value_the_baseline(interactions.index):
                # insert the empty value given in baseline into the values
                interactions[idx] = interactions.baseline_value
            else:  # manually set baseline to the empty value
                interactions.baseline_value = interactions[idx]
    # empty not in interactions but min_order is 0 (should be in the interactions)
    elif interactions.min_order == 0:
        # TODO(mmshlk): this might not be what we really want to do always: what if empty and baseline are different?
        # https://github.com/mmschlk/shapiq/issues/385
        interactions.interaction_lookup[()] = len(interactions.interaction_lookup)
        interactions.values = np.concatenate(
            (interactions.values, np.array([interactions.baseline_value])),
        )

    return interactions
