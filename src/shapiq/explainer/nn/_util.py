"""Utility function for the NormalKNNExplainer and the WeightedKNNExplainer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from shapiq.explainer.custom_types import ValidNNExplainerIndices

logger = logging.getLogger()


def warn_ignored_parameters(
    local_vars: Mapping[str, Any], ignored_parameter_names: Iterable[str], class_name: str
) -> None:
    for param in ignored_parameter_names:
        if local_vars[param] is not None:
            logger.warning(
                "A non-None value was passed as parameter `%s` to the constructor of %s, which will be ignored.",
                param,
                class_name,
            )


def assert_enough_training_samples(n_neighbors: int, n_samples_fit: int) -> None:
    """Check that the model was fitted on at least ``n_neighbors`` training samples, raise otherwise.

    A k-nearest neighbor model fitted on fewer training samples than ``n_neighbors`` cannot
    predict (scikit-learn raises the same error on ``predict``), so there is no model behavior
    to explain.

    Args:
        n_neighbors: The model's ``n_neighbors``.
        n_samples_fit: The number of training samples the model was fitted on.

    Raises:
        ValueError: If ``n_neighbors > n_samples_fit``.
    """
    if n_neighbors > n_samples_fit:
        msg = (
            f"Expected n_neighbors <= n_samples_fit, but got n_neighbors={n_neighbors} and "
            f"n_samples_fit={n_samples_fit}. A model fitted on fewer training samples than "
            "n_neighbors cannot predict; refit the model with n_neighbors <= n_samples_fit."
        )
        raise ValueError(msg)


def assert_valid_index_and_order(index: ValidNNExplainerIndices, max_order: int) -> None:
    """Check that the explainer index and max_order are valid for NN models, raise otherwise.

    The only valid index is ``'SV'``; the only valid max. order is ``1``.

    Args:
        index: The explainer index to validate.
        max_order: The max. order to validate.

    Raises:
        ValueError: If either of the parameters does not satisfy the requirements.
    """
    if index != "SV":
        msg = f"Explainer index '{index}' is invalid for nearest neighbor models. The only valid index is 'SV'."
        raise ValueError(msg)

    if max_order != 1:
        msg = f"Explanation order of {max_order} is invalid; the only valid order is 1."
        raise ValueError(msg)
