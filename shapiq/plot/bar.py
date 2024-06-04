"""Wrapper for the bar plot from the ``shap`` package."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

from ..utils.modules import check_import_module

__all__ = ["bar_plot"]


def bar_plot(
    list_of_interaction_values: list[InteractionValues],
    feature_names: Optional[np.ndarray] = None,
    show: bool = True,
    **kwargs,
):
    """Draws interaction values on a bar plot.

    Requires the ``shap`` Python package to be installed.

    Args:
        list_of_interaction_values: A list containing InteractionValues objects.
        feature_names: The feature names used for plotting. If no feature names are provided, the
            feature indices are used instead. Defaults to ``None``.
        feature_values: The feature values used for plotting. Defaults to ``None``.
        matplotlib: Whether to return a ``matplotlib`` figure. Defaults to ``True``.
        show: Whether ``matplotlib.pyplot.show()`` is called before returning. Default is ``True``.
            Setting this to ``False`` allows the plot to be customized further after it has been created.
        **kwargs: Keyword arguments passed to ``shap.plots.beeswarm()``.
    """
    check_import_module("shap")
    import shap

    assert len(np.unique([iv.max_order for iv in list_of_interaction_values])) == 1
    max_order = list_of_interaction_values[0].max_order

    _global_values = []
    _base_values = []
    _labels = []
    _first_iv = True
    for iv in list_of_interaction_values:
        _values_dict = {}
        for i in range(1, max_order + 1):
            _values_dict[i] = iv.get_n_order_values(i)
        _n_features = len(_values_dict[1])
        _shap_values = []
        for interaction in powerset(range(_n_features), min_size=1, max_size=max_order):
            _order = len(interaction)
            _values = _values_dict[_order]
            _shap_values.append(_values[interaction])
            if feature_names is not None and _first_iv:
                _labels.append(" x ".join(f"{feature_names[i]:0.10}" for i in interaction))
        _global_values.append(_shap_values)
        _base_values.append(iv.baseline_value)
        if _first_iv:
            _first_iv = False

    explanation = shap.Explanation(
        values=np.stack(_global_values),
        base_values=np.array(_base_values),
        feature_names=np.array(_labels) if feature_names is not None else None,
    )

    if show:
        ax = shap.plots.bar(explanation, **kwargs, show=False)
        ax.set_xlabel("mean(|Shapley Interaction value|)")
        plt.show()
    else:
        ax = shap.plots.bar(explanation, **kwargs, show=False)
        ax.set_xlabel("mean(|Shapley Interaction value|)")
        return ax
