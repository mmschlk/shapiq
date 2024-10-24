"""This utility module contains helper functions for plotting."""

import copy
from collections.abc import Iterable
from typing import Optional

import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

__all__ = ["get_interaction_values_and_feature_names", "abbreviate_feature_names"]


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
    feature_names = copy.deepcopy(feature_names)
    if feature_names is not None:
        feature_names = abbreviate_feature_names(feature_names)
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
            _name = " x ".join(str(feature_names[i]) for i in interaction)
        else:
            _name = " x ".join(f"{feature}" for feature in interaction)
        if feature_values is not None:
            _name += "\n"
            _name += " x ".join(f"{feature_values[i]}".strip()[0:4] for i in interaction)
        _labels.append(_name)
    _shap_values = np.array(_shap_values)
    _labels = np.array(_labels)
    return _shap_values, _labels


def abbreviate_feature_names(feature_names: Iterable[str]) -> list[str]:
    """A rudimentary function to abbreviate feature names for plotting.

    Args:
        feature_names: The feature names to be abbreviated.

    Returns:
        list[str]: The abbreviated feature names.
    """
    abbreviated_names = []
    for name in feature_names:
        name = str(name)
        name = name.strip()
        capital_letters = sum(1 for c in name if c.isupper())
        seperator_chars = (" ", "_", "-", ".")
        is_seperator_in_name = any([c in seperator_chars for c in name[:-1]])
        if is_seperator_in_name:
            for seperator in seperator_chars:
                name = name.replace(seperator, ".")
            name_parts = name.split(".")
            new_name = ""
            for part in name_parts:
                if part:
                    new_name += part[0].upper()
            abbreviated_names.append(new_name)
        elif capital_letters > 1:
            new_name = "".join([c for c in name if c.isupper()])
            abbreviated_names.append(new_name[0:3])
        else:
            abbreviated_names.append(name.strip()[0:3] + ".")
    return abbreviated_names
