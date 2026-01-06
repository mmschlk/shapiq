"""Utility function for the NormalKNNExplainer and the WeightedKNNExplainer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy.typing as npt

    from shapiq.explainer.custom_types import ExplainerIndices

logger = logging.getLogger()


def keep_first_n(mask: npt.NDArray[np.bool], n: int) -> npt.NDArray[np.bool]:
    """Sets all entries of the input array to False except the first ``n`` entries with value ``True``.

    This will just return a reference to the input array if ``np.sum(mask) <= n``

    Args:
        mask: The mask in question.
        n: The maximum number of true entries.
    """
    if n == 0:
        return np.zeros_like(mask)

    n_true = 0
    for i, val in enumerate(mask):
        n_true += int(val)
        if n_true == n:
            out = np.zeros_like(mask)
            out[: i + 1] = mask[: i + 1]
            return out

    return mask


def warn_ignored_parameters(
    local_vars: Mapping[str, Any], ignored_parameter_names: Iterable[str], class_name: str
) -> None:
    for param in ignored_parameter_names:
        if local_vars[param] is not None:
            logger.warning(
                "A non-None value was passed as parameter `%s` to the constructor of %s, which will be ignored.",
                class_name,
                param,
            )


def assert_valid_index_and_order(index: ExplainerIndices, max_order: int) -> None:
    """Check that the explainer index and max_order are valid for NN models, raise otherwise.

    The only valid indices are ``'SV'`` and ``'k-SII'``; the only valid max. order is ``1``.

    Args:
        index: The explainer index to validate.
        max_order: The max. order to validate.

    Raises:
        ValueError: If either of the parameters does not satisfy the requirements.
    """
    valid_indices: list[ExplainerIndices] = ["SV", "k-SII"]
    if index not in valid_indices:
        msg = f"Explainer index '{index}' is invalid for nearest-neighbor models. Valid indices are: {', '.join(valid_indices)}"
        raise ValueError(msg)

    if max_order != 1:
        msg = f"Explanation order of {max_order} is invalid; the only valid order is 1."
        raise ValueError(msg)
