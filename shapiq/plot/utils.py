"""This utility module contains helper functions for plotting."""

from typing import Optional

import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

__all__ = ["get_interaction_values_and_feature_names"]


def get_interaction_values_and_feature_names(
    interaction_values: InteractionValues,
    feature_names: Optional[np.ndarray] = None,
    feature_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Converts higher-order interaction values to SHAP-like vectors with associated labels.

    Args:
        interaction_values: The interaction values as an interaction object.
        feature_names: The feature names used for plotting. If no feature names are provided, the
            feature indices are used instead. Defaults to ``None``.
        feature_values: The feature values used for plotting. Defaults to ``None``.

    Returns:
        A tuple containing the SHAP values and the corresponding labels.
    """
    _values_dict = {}
    for i in range(1, interaction_values.max_order + 1):
        _values_dict[i] = interaction_values.get_n_order_values(i)
    _n_features = len(_values_dict[1])
    _shap_values = []
    _labels = []
    for interaction in powerset(
        range(_n_features), min_size=1, max_size=interaction_values.max_order
    ):
        _order = len(interaction)
        _values = _values_dict[_order]
        _shap_values.append(_values[interaction])
        if feature_names is not None:
            _name = " x ".join(f"{feature_names[i]}".strip()[0:4] + "." for i in interaction)
        else:
            _name = " x ".join(f"{feature}" for feature in interaction)
        if feature_values is not None:
            _name += "\n"
            _name += " x ".join(f"{feature_values[i]}".strip()[0:4] for i in interaction)
        _labels.append(_name)
    _shap_values = np.array(_shap_values)
    _labels = np.array(_labels)
    return _shap_values, _labels
