"""Validation module for explainer classes.

Validator functions are used to validate the input parameters of the explainer classes and provide
useful warnings or default values if necessary.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from shapiq.game_theory.indices import index_generalizes_bv, index_generalizes_sv, is_index_valid

if TYPE_CHECKING:
    from collections.abc import Callable

    from .custom_types import ExplainerIndices


def validate_data_predict_function(
    data: np.ndarray,
    predict_function: Callable[[np.ndarray], np.ndarray],
    *,
    raise_error: bool = False,
) -> None:
    """Validate the data for compatibility with the model.

    Args:
        data: A 2-dimensional matrix of inputs to be explained.

        predict_function: A callable function that takes data points as input and returns
            1-dimensional predictions. If provided, it will be used to validate the data.
            Defaults to ``None`` which means the model's predict function will be used.

        raise_error: Whether to raise an error if the data is not compatible with the model or
            only print a warning. Defaults to ``False``.

    Raises:
        TypeError: If the data is not a NumPy array.

    """
    message = ""

    # check input data type
    if not isinstance(data, np.ndarray):
        message += " The `data` must be a NumPy array."

    try:
        data_to_pred = data[0:1, :]
    except Exception as e:
        message += " The `data` must have at least one sample and be 2-dimensional."
        raise TypeError(message) from e

    try:
        pred = predict_function(data_to_pred)
    except Exception as e:
        message += f" The model's prediction failed with the following error: {e}."
        raise TypeError(message) from e

    if isinstance(pred, np.ndarray):
        if len(pred.shape) != 1:
            message += " The model's prediction must be a 1-dimensional array."
    else:
        message += " The model's prediction must be a NumPy array."

    if message != "":
        message = "The `data` and the model must be compatible." + message
        if raise_error:
            raise TypeError(message)
        warnings.warn(message, stacklevel=2)


def validate_index_and_max_order(
    index: ExplainerIndices,
    max_order: int,
) -> tuple[ExplainerIndices, int]:
    """Validate the index and max_order combination.

    Args:
        index: The index to be used.
        max_order: The maximum order of the index.

    Returns:
        A tuple containing the validated index and max_order.
    """
    validated_index = validate_index(index, max_order)
    validated_max_order = validate_max_order(validated_index, max_order)
    return validated_index, validated_max_order


def validate_index(index: ExplainerIndices, max_order: int) -> ExplainerIndices:
    """Validate the index and max_order combination.

    Args:
        index: The index to be validated.
        max_order: The maximum order of the index.

    Returns:
        The validated index.

    Raises:
        ValueError: If the index is not valid.
        Warning: If the index generalizes to SV or BV and max_order is 1.
    """
    is_index_valid(index, raise_error=True)
    msg = f"Mismatch between max_order={max_order} and index={index}. "
    if max_order == 1 and index_generalizes_sv(index):
        msg += f"{index} generalizes 'SV'. Setting index to 'SV'."
        warnings.warn(msg, stacklevel=2)
        return "SV"
    if max_order == 1 and index_generalizes_bv(index):
        msg += f"{index} generalizes 'BV'. Setting index to 'BV'."
        warnings.warn(msg, stacklevel=2)
        return "BV"
    return index


def validate_max_order(index: ExplainerIndices, max_order: int) -> int:
    """Validate the max_order for the selected index.

    Args:
        index: The index to be used.
        max_order: The maximum order to be validated.

    Returns:
        The validated max_order.
    """
    msg = f"Mismatch between max_order={max_order} and index={index}. Setting max_order=1. "
    if max_order > 1 and index == "SV":
        warnings.warn(msg, stacklevel=2)
        return 1
    if max_order > 1 and index == "BV":
        warnings.warn(msg, stacklevel=2)
        return 1
    return max_order
