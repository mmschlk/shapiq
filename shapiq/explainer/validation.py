"""This module contains the validator functions for the explainer module.

Validator functions are used to validate the input parameters of the explainer classes and provide
useful warnings or default values if necessary.
"""

import warnings
from collections.abc import Callable

import numpy as np

from ..game_theory.indices import index_generalizes_bv, index_generalizes_sv


def validate_data(
    data: np.ndarray,
    predict_function: Callable[[np.ndarray], np.ndarray] | None = None,
    raise_error: bool = False,
) -> None:
    """Validate the data for compatibility with the model.

    Args:
        data: A 2-dimensional matrix of inputs to be explained.
        predict_function: The model's prediction function.
        raise_error: Whether to raise an error if the data is not compatible with the model or
            only print a warning. Defaults to ``False``.

    Raises:
        TypeError: If the data is not a NumPy array.
    """
    message = "The `data` and the model must be compatible."
    if not isinstance(data, np.ndarray):
        message += " The `data` must be a NumPy array."
        raise TypeError(message)
    if predict_function is not None:
        try:
            # TODO (mmschlk): This can take a long time for large datasets and slow models
            pred = predict_function(data)
            if isinstance(pred, np.ndarray):
                if len(pred.shape) > 1:
                    message += " The model's prediction must be a 1-dimensional array."
                    raise ValueError()
            else:
                message += " The model's prediction must be a NumPy array."
                raise ValueError()
        except Exception as e:
            if raise_error:
                raise ValueError(message) from e
            else:
                warnings.warn(message)


def set_random_state(random_state: int | None, object_with_rng: object) -> None:
    """Sets the random state for all rng objects in the explainer.

    Args:
        random_state: The random state to re-initialize, Explainer, Imputer and Approximator with.
            Defaults to ``None`` which does not change the random state.
        object: The object to set the random state for
    """
    # TODO: writhe semantic test for this method
    if random_state is not None:
        if hasattr(object_with_rng, "_rng"):  # default attribute
            object_with_rng._rng = np.random.default_rng(random_state)
        # explainer can have an imputer
        if hasattr(object_with_rng, "_imputer"):
            object_with_rng._imputer._rng = np.random.default_rng(random_state)
        # explainer can have an approximator
        if hasattr(object_with_rng, "_approximator"):
            object_with_rng._approximator._rng = np.random.default_rng(random_state)
            # approximators inside an explainer can have a sampler
            if hasattr(object_with_rng._approximator, "_sampler"):
                object_with_rng._approximator._sampler._rng = np.random.default_rng(random_state)
        # appoximators can have a sampler
        if hasattr(object_with_rng, "_sampler"):
            object_with_rng._sampler._rng = np.random.default_rng(random_state)


def validate_budget(budget: int | None, n_players: int) -> int:
    """Validate the budget parameter.

    Args:
        budget: The budget to be used.
        n_players: The number of players in the game.

    Returns:
        The validated budget.
    """
    if budget is None:
        budget = 2**n_players
        if budget > 2048:
            warnings.warn(
                f"Using the budget of 2**n_features={budget}, which might take long\
                          to compute. Set the `budget` parameter to suppress this warning."
            )
    return budget


def validate_index(index: str, max_order: int) -> str:
    """Validate the index and max_order combination.

    Args:
        index: The index to be used.
        max_order: The maximum order of the index.

    Returns:
        The validated index.
    """
    if max_order == 1 and index_generalizes_sv(index):
        warnings.warn(
            f"`max_order=1` but index `{index}` generalizes `SV`, setting `index = 'SV'`."
        )
        return "SV"
    if max_order == 1 and index_generalizes_bv(index):
        warnings.warn(
            f"`max_order=1` but index `{index}` generalizes `BV`, setting `index = 'BV'`."
        )
        return "BV"
    return index


def validate_max_order(index: str, max_order: int) -> int:
    """Validate the max_order for the selected index.

    Args:
        index: The index to be used.
        max_order: The maximum order of the index.

    Returns:
        The validated max_order.
    """
    if max_order > 1 and index == "SV":
        warnings.warn(f"`max_order > 1` but index `{index}` is `SV`, setting `max_order = 1`.")
        return 1
    if max_order > 1 and index == "BV":
        warnings.warn(f"`max_order > 1` but index `{index}` is `BV`, setting `max_order = 1`.")
        return 1
    return max_order
