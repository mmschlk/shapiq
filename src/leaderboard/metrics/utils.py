"""Utility functions for metric computation."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    import numpy as np


def remove_empty_value_if_needed(
    value: InteractionValues | np.ndarray,
) -> InteractionValues | np.ndarray:
    """Set the empty interaction value to zero for InteractionValues inputs.

    NumPy arrays are returned unchanged.
    """
    if not isinstance(value, InteractionValues):
        return value

    try:
        new_value = copy.deepcopy(value)
        empty_index = new_value.interaction_lookup[()]
        new_value.values[empty_index] = 0
    except KeyError:
        return value
    else:
        return new_value
