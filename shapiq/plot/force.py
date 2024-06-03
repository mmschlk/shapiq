"""Wrapper for the force plot from the shap package."""

from typing import Optional
import numpy as np
import warnings

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

from ..utils.modules import check_import_module

__all__ = ["force_plot"]


def force_plot(
    interaction_values: InteractionValues,
    feature_values: Optional[np.ndarray] = None,
    feature_names: Optional[np.ndarray] = None,
    **kwargs
):
        """Draws interaction values on a force plot.

        Requires the ``shap`` Python package to be installed.

        Args:
            interaction_values: The interaction values as an interaction object.
            feature_names: The feature names used for plotting. If no feature names are provided, the
                feature indices are used instead. Defaults to ``None``.
            feature_values: The feature values used for plotting. Defaults to ``None``.
            **kwargs: Keyword arguments passed to ``shap.plots.force()``.
        """
        check_import_module("shap")
        import shap
        if interaction_values.max_order == 1:
            return shap.plots.force(
                    base_value=interaction_values.baseline_value,
                    shap_values=interaction_values.get_n_order_values(1),
                    features=feature_values,
                    feature_names=feature_names,
                    **kwargs
                )
        else:
            _values_dict = {}
            for i in range(1, interaction_values.max_order + 1):
                 _values_dict[i] = interaction_values.get_n_order_values(i)
            _n_features = len(_values_dict[1])
            _shap_values = []
            _features = []
            _feature_names = []
            for interaction in powerset(range(_n_features), min_size=1, max_size=interaction_values.max_order):
                _order = len(interaction)
                _values = _values_dict[_order]
                _shap_values.append(_values[interaction])
                if feature_values is not None:
                    _features.append(' x '.join(f'{feature_values[i]:0.3}' for i in interaction))
                if feature_names is not None:
                    _feature_names.append(' x '.join(f'{feature_names[i]:0.5}' for i in interaction))

            return shap.plots.force(
                    base_value=interaction_values.baseline_value,
                    shap_values=np.array(_shap_values),
                    features=np.array(_features) if feature_values is not None else None,
                    feature_names=np.array(_feature_names) if feature_names is not None else None,
                    **kwargs
                )  