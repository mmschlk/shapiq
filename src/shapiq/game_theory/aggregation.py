"""Aggregation functions for summarizing base interaction indices into efficient indices useful for explanations."""

from __future__ import annotations

import warnings

import numpy as np
import scipy as sp

from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset


def _change_index(index: str) -> str:
    """Changes the index of the interaction values to the new index.

    Args:
        index: The current index of the interaction values.

    Returns:
        The new index of the interaction values.

    """
    if index in ["SV", "BV"]:  # no change for probabilistic values like SV or BV
        return index
    return f"k-{index}"


def aggregate_base_attributions(
    interactions: dict[tuple[int, ...], float],
    index: str,
    order: int,
    min_order: int,
    baseline_value: float,
) -> tuple[dict[tuple[int, ...], float], str, int]:
    """Aggregates the interactions into an efficient interactions.

    An example aggregation would be the transformation from `SII` values to `k-SII` values.

    Args:
        interactions: The base interaction values to aggregate.
        index: The index of the interaction values.
        order: The order of the aggregation. For example, the order of the k-SII aggregation.
        min_order: The minimum order of the base interactions. If the base interactions have a minimum
            order greater than 1, a warning is raised.
        baseline_value: The baseline value of the interaction values. For example, the baseline value
            of the SII values must not be the same as the values of the empty set.

    Returns:
        A tuple containing:
            - A dictionary mapping interactions to their values.
            - The new index of the interaction values.
            - The new minimum order of the interaction values (always 0 for this aggregation).

    Raises:
        ValueError: If the `order` is smaller than 0.
    """
    if min_order > 1:
        warnings.warn(
            UserWarning(
                "The base interaction values have a minimum order greater than 1. Aggregation may "
                "not be meaningful.",
            ),
            stacklevel=2,
        )

    bernoulli_numbers = sp.special.bernoulli(order)  # used for aggregation
    transformed_interactions: dict[tuple, float] = {(): baseline_value}  # storage
    # iterate over all interactions in base_interactions and project them onto all interactions T
    # where 1 <= |T| <= order
    for base_interaction, base_interaction_value in interactions.items():
        for interaction in powerset(base_interaction, min_size=1, max_size=order):
            scaling = float(bernoulli_numbers[len(base_interaction) - len(interaction)])
            update_interaction = scaling * base_interaction_value
            if update_interaction == 0:
                continue
            transformed_interactions[interaction] = (
                transformed_interactions.get(interaction, 0) + update_interaction
            )
            # if the interactions sum to 0, we pop them from the dict
            if transformed_interactions[interaction] == 0:
                transformed_interactions.pop(interaction)

    # update the index name after the aggregation (e.g., SII -> k-SII)
    new_index = _change_index(index)
    return (
        transformed_interactions,
        new_index,
        0,
    )  # always order 0 for this aggregation


def aggregate_base_interaction(
    base_interactions: InteractionValues,
    order: int | None = None,
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
        >>> k_sii_values = aggregate_base_interaction(sii_values)
        >>> k_sii_values.index
        'k-SII'
        >>> k_sii_values.baseline_value
        0
        >>> k_sii_values.interaction_lookup
        {(): 0, (1,): 1, (2,): 2, (3,): 3, (1, 2): 4, (2, 3): 5, (1, 3): 6}
        >>> k_sii_values.max_order
        2

    """
    order = order or base_interactions.max_order
    transformed_interactions, new_index, new_min_order = aggregate_base_attributions(
        interactions=base_interactions.interactions,
        index=base_interactions.index,
        order=order,
        min_order=base_interactions.min_order,
        baseline_value=float(base_interactions.baseline_value),
    )

    return InteractionValues(
        values=transformed_interactions,
        n_players=base_interactions.n_players,
        index=new_index,
        baseline_value=base_interactions.baseline_value,
        min_order=new_min_order,
        max_order=order,
        estimated=base_interactions.estimated,
        estimation_budget=base_interactions.estimation_budget,
    )


def aggregate_to_one_dimension(
    interactions: InteractionValues,
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens the higher-order interaction values to positive and negative one-dimensional values.

    The aggregation summarizes all higher-order interaction in the positive and negative
    one-dimensional values for each player. The aggregation is done by distributing the interaction
    scores uniformly to all players in the interaction. For example, the interaction value 5 of
    the interaction `(1, 2)` is distributed to player 1 and player 2 as 2.5 each.

    Args:
        interactions: The interaction values to convert.

    Returns:
        The positive and negative interaction values as a 1-dimensional array for each player.

    """
    n = interactions.n_players
    pos_values = np.zeros(shape=(n,), dtype=float)
    neg_values = np.zeros(shape=(n,), dtype=float)

    for interaction in interactions.interaction_lookup:
        if len(interaction) == 0:
            continue  # skip the empty set
        interaction_value = interactions[interaction] / len(interaction)  # distribute uniformly
        for player in interaction:
            if interaction_value >= 0:
                pos_values[player] += interaction_value
            else:
                neg_values[player] += interaction_value

    return pos_values, neg_values
