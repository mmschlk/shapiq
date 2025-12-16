"""Utility function for the NormalKNNExplainer and the WeightedKNNExplainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    import numpy.typing as npt


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


def interaction_values_from_array(
    shapley_values: npt.NDArray[np.floating],
) -> InteractionValues:
    """Convert an array of Shapley values to a ``shapiq.interaction_values.InteractionValues`` object.

    Args:
        shapley_values: An ``np.ndarray`` containing the Shapley value of the ith training point at index i.

    Returns:
        An ``InteractionValues`` object containing the provided Shapley values with an appropriate ``interaction_lookup`` dict and with ``min_order == max_order == 1`` set.
    """
    n_players = shapley_values.shape[0]
    interaction_lookup: dict[tuple[int, ...], int] = {(i,): i for i in range(n_players)}

    return InteractionValues(
        shapley_values,
        index="SV",
        min_order=1,
        max_order=1,
        n_players=n_players,
        baseline_value=0,
        interaction_lookup=interaction_lookup,
    )


def interaction_values_to_array(
    interaction_values: InteractionValues,
) -> npt.NDArray[np.floating]:
    """Extract an array of Shapley values from a ``shapiq.interaction_values.InteractionValues`` object.

    Args:
        interaction_values: An InteractionValues object with ``max_order==1``

    Returns:
        An ``np.ndarray`` of shape ``(n_players,)`` containing at index i the Shapley value of player i.
    """
    if interaction_values.max_order != 1:
        msg = f"Max order must be 1 but was {interaction_values.max_order}"
        raise ValueError(msg)

    out = np.zeros((interaction_values.n_players,))

    for coalition, lookup_idx in interaction_values.interaction_lookup.items():
        if coalition == ():
            continue
        out[coalition[0]] = interaction_values.values[lookup_idx]

    return out
