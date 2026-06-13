"""Utility functions for metric computation."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from collections.abc import Iterable


def remove_empty_value_if_needed(
    value: InteractionValues | np.ndarray,
) -> InteractionValues | np.ndarray:
    """Set the empty interaction value to zero for InteractionValues inputs.

    NumPy arrays are returned unchanged.
    """
    if not isinstance(value, InteractionValues):
        return value

    try:
        value.interaction_lookup[()]
    except KeyError:
        return value

    new_value = copy.deepcopy(value)
    new_value.interactions[()] = 0.0
    return new_value


def prepare_metric_inputs(ground_truth: object, estimated: object) -> tuple[np.ndarray, np.ndarray]:
    """Prepare metric inputs while preserving existing NumPy-array behavior.

    InteractionValues are aligned by the interaction keys they actually store.
    The empty interaction key is ignored, and missing interactions in either
    input are represented as 0.0. This avoids enumerating the full powerset.
    """
    if isinstance(ground_truth, InteractionValues) and isinstance(estimated, InteractionValues):
        return _prepare_interaction_values(ground_truth, estimated)

    ground_truth_array = np.asarray(ground_truth, dtype=float)
    estimated_array = np.asarray(estimated, dtype=float)

    if ground_truth_array.shape != estimated_array.shape:
        msg = "ground_truth and estimated must have the same shape"
        raise ValueError(msg)

    return ground_truth_array, estimated_array


def _prepare_interaction_values(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
) -> tuple[np.ndarray, np.ndarray]:
    interactions = sorted(
        _non_empty_keys(ground_truth.interaction_lookup)
        | _non_empty_keys(estimated.interaction_lookup),
        key=lambda interaction: (len(interaction), interaction),
    )

    return (
        _values_for_interactions(ground_truth, interactions),
        _values_for_interactions(estimated, interactions),
    )


def _non_empty_keys(interaction_lookup: dict[tuple[int, ...], int]) -> set[tuple[int, ...]]:
    return {interaction for interaction in interaction_lookup if interaction != ()}


def _values_for_interactions(
    values: InteractionValues,
    interactions: Iterable[tuple[int, ...]],
) -> np.ndarray:
    return np.array(
        [
            values.values[values.interaction_lookup[interaction]]
            if interaction in values.interaction_lookup
            else 0.0
            for interaction in interactions
        ],
        dtype=float,
    )
