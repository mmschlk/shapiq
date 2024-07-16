"""Wrapper for the waterfall plot from the ``shap`` package."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.utils.modules import check_import_module

from .utils import get_interaction_values_and_feature_names

__all__ = ["waterfall_plot"]


def waterfall_plot(
    interaction_values: InteractionValues,
    feature_names: Optional[np.ndarray] = None,
    feature_values: Optional[np.ndarray] = None,
    show: bool = False,
    max_display: int = 10,
) -> Optional[plt.Axes]:
    """Draws interaction values on a waterfall plot.

    Note:
        Requires the ``shap`` Python package to be installed.

    Args:
        interaction_values: The interaction values as an interaction object.
        feature_names: The feature names used for plotting. If no feature names are provided, the
            feature indices are used instead. Defaults to ``None``.
        feature_values: The feature values used for plotting. Defaults to ``None``.
        show: Whether to show the plot. Defaults to ``False``.
        max_display: The maximum number of interactions to display. Defaults to ``10``.
    """
    check_import_module("shap")
    import shap

    if interaction_values.max_order == 1:
        shap_explanation = shap.Explanation(
            values=interaction_values.get_n_order_values(1),
            base_values=interaction_values.baseline_value,
            data=feature_values,
            feature_names=feature_names,
        )
    else:
        _shap_values, _labels = get_interaction_values_and_feature_names(
            interaction_values, feature_names, feature_values
        )

        shap_explanation = shap.Explanation(
            values=np.array(_shap_values),
            base_values=np.array([interaction_values.baseline_value], dtype=float),
            data=None,
            feature_names=_labels,
        )

    return shap.plots.waterfall(shap_explanation, max_display=max_display, show=show)
