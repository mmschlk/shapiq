"""Wrapper for the force plot from the ``shap`` package."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.utils.modules import check_import_module

from .utils import get_interaction_values_and_feature_names

__all__ = ["force_plot"]


def force_plot(
    interaction_values: InteractionValues,
    feature_names: Optional[np.ndarray] = None,
    feature_values: Optional[np.ndarray] = None,
    matplotlib: bool = True,
    show: bool = False,
    **kwargs,
) -> Optional[plt.Figure]:
    """Draws interaction values on a force plot.

    Requires the ``shap`` Python package to be installed.

    Args:
        interaction_values: The interaction values as an interaction object.
        feature_names: The feature names used for plotting. If no feature names are provided, the
            feature indices are used instead. Defaults to ``None``.
        feature_values: The feature values used for plotting. Defaults to ``None``.
        matplotlib: Whether to return a ``matplotlib`` figure. Defaults to ``True``.
        show: Whether to show the plot. Defaults to ``False``.
        **kwargs: Keyword arguments passed to ``shap.plots.force()``.
    """
    check_import_module("shap")
    import shap

    _shap_values, _labels = get_interaction_values_and_feature_names(
        interaction_values, feature_names, feature_values
    )

    return shap.plots.force(
        base_value=np.array([interaction_values.baseline_value], dtype=float),  # must be array
        shap_values=np.array(_shap_values),
        feature_names=_labels,
        matplotlib=matplotlib,
        show=show,
        **kwargs,
    )
