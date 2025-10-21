"""This module contains all custom types used in the shapiq package."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatVector = NDArray[np.floating]
"""A 1D array of floating point numbers, typically used to represent game values or"""

IntVector = NDArray[np.integer]
"""A 1D array of integers, typically used to represent player indices or counts."""

BoolVector = NDArray[np.bool_]
"""A 1D array of boolean values, typically used to represent the presence or absence of players in
a coalition."""

CoalitionMatrix = NDArray[np.bool_]
"""A 2D one-hot encoded matrix representing coalitions. A 1 denotes a player is part of the
coalition, and a 0 denotes they are not. The array is of shape ``(n_coalitions, n_players)``."""

CoalitionTuple = tuple[int, ...]
"""A tuple representing a coalition of players. Each integer is a player index, and the tuple is
sorted in ascending order."""

CoalitionsTuples = Collection[CoalitionTuple]
"""A list of coalitions, where each coalition is represented as a tuple of player indices."""

InteractionTuple = tuple[int, ...]
"""A tuple representing an interaction of players. Each integer is a player index, and the tuple
is sorted in ascending order."""

InteractionScores = dict[InteractionTuple, float]
"""A dictionary mapping interactions (as tuples of player indices) to their corresponding
interaction score."""

GameScores = dict[CoalitionTuple, float]
"""A dictionary mapping coalitions (as tuples of player indices) to their corresponding game
value."""

GameValues = FloatVector
"""A 1D array representing the values of coalitions in a game. The array is of shape
``(n_coalitions,)``, where each entry corresponds to output of a game evaluation for a coalition."""

Model = Any
"""A generic type denoting a machine learning model."""


@runtime_checkable
class SklearnLikeModel(Protocol):
    """A protocol representing a scikit-learn-like model."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> Any: ...
    def predict(self, X: ArrayLike) -> np.ndarray: ...


IndexType = Literal[
    "SII",
    "BII",
    "CHII",
    "Co-Moebius",
    "SGV",
    "BGV",
    "CHGV",
    "IGV",
    "EGV",
    "k-SII",
    "STII",
    "FSII",
    "kADD-SHAP",
    "FBII",
    "SV",
    "BV",
    "JointSV",
    "Moebius",
    "ELC",
]
"""A type representing the indices used throughout the package."""


JSONPrimitive = str | int | float | bool | None
JSONType = JSONPrimitive | Sequence["JSONType"] | Mapping[str, "JSONType"]


class MetadataBlock(TypedDict):
    """Metadata block for saving objects as json."""

    object_name: str
    data_type: Literal["interaction_values", "game"] | None
    version: str
    timestamp: str
    created_from: str | None
    description: str | None
    parameters: JSONType


NumericArray = NDArray[np.number]
