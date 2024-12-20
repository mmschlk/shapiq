"""Aggregation functions for summarizing base interaction indices into
efficient indices useful for explanations"""

import warnings
from typing import Optional

import numpy as np
import scipy as sp

from ..interaction_values import InteractionValues
from ..utils.sets import powerset


def _change_index(index: str) -> str:
    """Changes the index of the interaction values to the new index.

    Args:
        index: The current index of the interaction values.

    Returns:
        The new index of the interaction values.
    """
    if index in ["SV", "BV"]:  # no change for probabilistic values like SV or BV
        return index
    new_index = "-".join(("k", index))
    return new_index


def aggregate_interaction_values(
    base_interactions: InteractionValues, order: Optional[int] = None
) -> InteractionValues:
    """Aggregates the basis interaction values into an efficient interaction index.

    An example aggregation would be the transformation from `SII` values to `k-SII` values.

    Args:
        base_interactions: The basis interaction values to aggregate.
        order: The order of the aggregation. For example, the order of the k-SII aggregation. If
            `None`, the maximum order of the base interactions is used. Defaults to `None`.

    Returns:
        The aggregated interaction values.

    Raises:
        ValueError: If the `order` is smaller than 0.

    Examples:
        >>> import numpy as np
        >>> from shapiq.interaction_values import InteractionValues
        >>> sii_values = InteractionValues(
        ...     n_players=3,
        ...     values=np.array([-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ...     index="SII",
        ...     interaction_lookup={(): 0, (1,): 1, (2,): 2, (3,): 3, (1, 2): 4, (2, 3): 5, (1, 3): 6},
        ...     baseline_value=0,  # for SII, the baseline value must not be the same as the values of emptyset
        ...     min_order=0,
        ...     max_order=2,
        ... )
        >>> k_sii_values = aggregate_interaction_values(sii_values)
        >>> k_sii_values.index
        'k-SII'
        >>> k_sii_values.baseline_value
        0
        >>> k_sii_values.interaction_lookup
        {(): 0, (1,): 1, (2,): 2, (3,): 3, (1, 2): 4, (2, 3): 5, (1, 3): 6}
        >>> k_sii_values.max_order
        2
    """
    # sanitize input parameters
    order = order or base_interactions.max_order

    if base_interactions.min_order > 1:
        warnings.warn(
            UserWarning(
                "The base interaction values have a minimum order greater than 1. Aggregation may "
                "not be meaningful."
            )
        )

    bernoulli_numbers = sp.special.bernoulli(order)  # used for aggregation
    baseline_value = base_interactions.baseline_value
    transformed_dict: dict[tuple, float] = {tuple(): baseline_value}  # storage
    # iterate over all interactions in base_interactions and project them onto all interactions T
    # where 1 <= |T| <= order
    for base_interaction, pos in base_interactions.interaction_lookup.items():
        base_interaction_value = float(base_interactions.values[pos])
        for interaction in powerset(base_interaction, min_size=1, max_size=order):
            scaling = float(bernoulli_numbers[len(base_interaction) - len(interaction)])
            update_interaction = scaling * base_interaction_value
            try:
                transformed_dict[interaction] += update_interaction
            except KeyError:
                transformed_dict[interaction] = update_interaction

    lookup: dict[tuple[int, ...], int] = {}  # maps interactions to their index in the values vector
    aggregated_values = np.zeros(len(transformed_dict), dtype=float)
    for pos, (interaction, interaction_value) in enumerate(transformed_dict.items()):
        lookup[interaction] = pos
        aggregated_values[pos] = interaction_value

    # update the index name after the aggregation (e.g., SII -> k-SII)
    new_index = _change_index(base_interactions.index)

    return InteractionValues(
        n_players=base_interactions.n_players,
        values=aggregated_values,
        index=new_index,
        interaction_lookup=lookup,
        baseline_value=baseline_value,
        min_order=0,  # always order 0 for this aggregation
        max_order=order,
        estimated=base_interactions.estimated,
        estimation_budget=base_interactions.estimation_budget,
    )


def aggregate_to_one_dimension(interactions: InteractionValues) -> tuple[np.ndarray, np.ndarray]:
    """Converts the interaction values to positive and negative one-dimensional values.

    Args:
        interactions: The interaction values to convert.

    Returns:
        The positive and negative one-dimensional values for each player. Both arrays have the shape
            `(n,)` where `n` is the number of players.
    """

    max_order = interactions.max_order
    min_order = max(interactions.min_order, 1)
    n = interactions.n_players

    pos_values = np.zeros(shape=(n,), dtype=float)
    neg_values = np.zeros(shape=(n,), dtype=float)

    for interaction in powerset(set(range(n)), min_size=min_order, max_size=max_order):
        interaction_value = interactions[interaction] / len(interaction)  # distribute uniformly
        for player in interaction:
            if interaction_value >= 0:
                pos_values[player] += interaction_value
            else:
                neg_values[player] += interaction_value
    return pos_values, neg_values
