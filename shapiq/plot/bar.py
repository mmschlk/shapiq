"""Wrapper for the bar plot from the ``shap`` package."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..interaction_values import InteractionValues
from ..utils.modules import check_import_module
from .utils import get_interaction_values_and_feature_names

__all__ = ["bar_plot"]


def bar_plot(
    list_of_interaction_values: list[InteractionValues],
    feature_names: Optional[np.ndarray] = None,
    show: bool = False,
    **kwargs,
) -> Optional[plt.Axes]:
    """Draws interaction values on a bar plot.

    Requires the ``shap`` Python package to be installed.

    Args:
        list_of_interaction_values: A list containing InteractionValues objects.
        feature_names: The feature names used for plotting. If no feature names are provided, the
            feature indices are used instead. Defaults to ``None``.
        show: Whether ``matplotlib.pyplot.show()`` is called before returning. Default is ``True``.
            Setting this to ``False`` allows the plot to be customized further after it has been created.
        **kwargs: Keyword arguments passed to ``shap.plots.beeswarm()``.
    """
    check_import_module("shap")
    import shap

    assert len(np.unique([iv.max_order for iv in list_of_interaction_values])) == 1

    _global_values = []
    _base_values = []
    _labels = []
    _first_iv = True
    for iv in list_of_interaction_values:

        _shap_values, _names = get_interaction_values_and_feature_names(iv, feature_names, None)
        if _first_iv:
            _labels = _names
            _first_iv = False
        _global_values.append(_shap_values)
        _base_values.append(iv.baseline_value)

    _labels = np.array(_labels) if feature_names is not None else None
    explanation = shap.Explanation(
        values=np.stack(_global_values),
        base_values=np.array(_base_values),
        feature_names=_labels,
    )

    ax = shap.plots.bar(explanation, **kwargs, show=False)
    ax.set_xlabel("mean(|Shapley Interaction value|)")
    if not show:
        return ax
    plt.show()
