"""This module provides the functionality to transform SII values into k-SII values."""

from typing import Optional, Union

import numpy as np
from interaction_values import InteractionValues
from scipy.special import bernoulli

from shapiq.approximator._base import Approximator
from shapiq.utils import generate_interaction_lookup, powerset


class KShapleyMixin:
    """Mixin class for the computation of k-Shapley values from SII estimators.

    Provides the common functionality for SII-based approximators like `PermutationSamplingSII` or
    `ShapIQ` for SII to transform their interaction scores into nSII values. The k-SII values are
    proposed in this `paper<https://proceedings.mlr.press/v206/bordt23a>`_.
    """

    def transforms_sii_to_ksii(
        self: Approximator,
        sii_values: Union[np.ndarray[float], InteractionValues],
    ) -> Union[np.ndarray[float], InteractionValues]:
        """Transforms the SII values into k-SII values.

        Args:
            sii_values: The SII values to transform. Can be either a numpy array or an
                InteractionValues object. The output will be of the same type.

        Returns:
            The k-SII values in the same format as the input.
        """
        return transforms_sii_to_ksii(
            sii_values=sii_values,
            approximator=self,
        )


def transforms_sii_to_ksii(
    sii_values: Union[np.ndarray[float], InteractionValues],
    *,
    approximator: Optional[Approximator] = None,
    n: Optional[int] = None,
    max_order: Optional[int] = None,
    interaction_lookup: Optional[dict] = None,
) -> Union[np.ndarray[float], InteractionValues]:
    """Transforms the SII values into k-SII values.

    Args:
        sii_values: The SII values to transform. Can be either a numpy array or an
            InteractionValues object. The output will be of the same type.
        approximator: The approximator used to estimate the SII values. If provided, meta
            information for the transformation is taken from the approximator. Defaults to None.
        n: The number of players. Required if `approximator` is not provided. Defaults to None.
        max_order: The maximum order of the approximation. Required if `approximator` is not
            provided. Defaults to None.
        interaction_lookup: A dictionary that maps interactions to their index in the values
            vector. If `interaction_lookup` is not provided, it is computed from the `n_players`
            and the `max_order` parameters. Defaults to `None`.

    Returns:
        The k-SII values in the same format as the input.
    """
    if isinstance(sii_values, InteractionValues):
        ksii_values = _calculate_ksii_from_sii(
            sii_values.values,
            sii_values.n_players,
            sii_values.max_order,
            sii_values.interaction_lookup,
        )
        return InteractionValues(
            values=ksii_values,
            index="k-SII",
            max_order=sii_values.max_order,
            min_order=sii_values.min_order,
            n_players=sii_values.n_players,
            interaction_lookup=sii_values.interaction_lookup,
            estimated=sii_values.estimated,
            estimation_budget=sii_values.estimation_budget,
        )
    elif approximator is not None:
        return _calculate_ksii_from_sii(
            sii_values, approximator.n, approximator.max_order, approximator.interaction_lookup
        )
    elif n is not None and max_order is not None:
        if interaction_lookup is None:
            interaction_lookup = generate_interaction_lookup(n, 1, max_order)
        return _calculate_ksii_from_sii(sii_values, n, max_order, interaction_lookup)
    else:
        raise ValueError(
            "If the SII values are not provided as InteractionValues, the approximator "
            "or the number of players and the maximum order of the approximation must be "
            "provided."
        )


def _calculate_ksii_from_sii(
    sii_values: np.ndarray[float],
    n: int,
    max_order: int,
    interaction_lookup: Optional[dict] = None,
) -> np.ndarray[float]:
    """Calculates the k-SII values from the SII values.

    Args:
        sii_values: The SII values to transform.
        n: The number of players.
        max_order: The maximum order of the approximation.
        interaction_lookup: A dictionary that maps interactions to their index in the values
            vector. If `interaction_lookup` is not provided, it is computed from the `n_players`,
            `min_order`, and `max_order` parameters. Defaults to `None`.

    Returns:
        The nSII values.
    """
    # compute nSII values from SII values
    bernoulli_numbers = bernoulli(max_order)
    nsii_values = np.zeros_like(sii_values)
    # all subsets S with 1 <= |S| <= max_order
    for subset in powerset(set(range(n)), min_size=1, max_size=max_order):
        interaction_size = len(subset)
        try:
            interaction_index = interaction_lookup[subset]
            ksii_value = sii_values[interaction_index]
        except KeyError:
            continue  # a zero value is not scaled # TODO: verify this
        # go over all subsets T of length |S| + 1, ..., n that contain S
        for T in powerset(set(range(n)), min_size=interaction_size + 1, max_size=max_order):
            if set(subset).issubset(T):
                effect_index = interaction_lookup[T]  # get the index of T
                effect_value = sii_values[effect_index]  # get the effect of T
                bernoulli_factor = bernoulli_numbers[len(T) - interaction_size]
                ksii_value += bernoulli_factor * effect_value
        nsii_values[interaction_index] = ksii_value
    return nsii_values


def convert_ksii_into_one_dimension(
    ksii_values: InteractionValues,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Converts the k-SII values into one-dimensional values.

    Args:
        ksii_values: The k-SII values to convert.

    Returns:
        The positive and negative one-dimensional values.
    """
    if ksii_values.index != "k-SII":
        raise ValueError(
            "Only nSII values can be converted into one-dimensional k-SII values. Please use the "
            "transforms_sii_to_ksii method to convert SII values into k-SII values."
        )
    max_order = ksii_values.max_order
    min_order = ksii_values.min_order
    n = ksii_values.n_players

    pos_ksii_values = np.zeros(shape=(n,), dtype=float)
    neg_ksii_values = np.zeros(shape=(n,), dtype=float)

    for subset in powerset(set(range(n)), min_size=min_order, max_size=max_order):
        ksii_value = ksii_values[subset] / len(subset)  # distribute uniformly
        for player in subset:
            if ksii_value >= 0:
                pos_ksii_values[player] += ksii_value
            else:
                neg_ksii_values[player] += ksii_value
    return pos_ksii_values, neg_ksii_values
