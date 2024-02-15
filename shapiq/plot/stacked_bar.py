"""This module contains functions to plot the n_sii stacked bar charts."""

__all__ = ["stacked_bar_plot"]

from copy import deepcopy
from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from ._config import COLORS_N_SII


def stacked_bar_plot(
    feature_names: Union[list, np.ndarray],
    n_shapley_values_pos: dict,
    n_shapley_values_neg: dict,
    n_sii_max_order: Optional[int] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Plot the n-SII values for a given instance.

    This stacked bar plot can be used to visualize the amount of interaction between the features
    for a given instance. The n-SII values are plotted as stacked bars with positive and negative
    parts stacked on top of each other. The colors represent the order of the n-SII values. For a
    detailed explanation of this plot, see this `research paper <https://proceedings.mlr.press/v206/bordt23a/bordt23a.pdf>`_.

    An example of the plot is shown below.

    .. image:: /_static/stacked_bar_exampl.png
        :width: 400
        :align: center

    Args:
        feature_names (list): The names of the features.
        n_shapley_values_pos (dict): The positive n-SII values.
        n_shapley_values_neg (dict): The negative n-SII values.
        n_sii_max_order (int): The order of the n-SII values.
        title (str): The title of the plot.
        xlabel (str): The label of the x-axis.
        ylabel (str): The label of the y-axis.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing the figure and
            the axis of the plot.

    Note:
        To change the figure size, font size, etc., use the [matplotlib parameters](https://matplotlib.org/stable/users/explain/customizing.html).

    Example:
        >>> import numpy as np
        >>> from shapiq.plot import stacked_bar_plot
        >>> n_shapley_values_pos = {
        ...     1: np.asarray([1, 0, 1.75]),
        ...     2: np.asarray([0.25, 0.5, 0.75]),
        ...     3: np.asarray([0.5, 0.25, 0.25]),
        ... }
        >>> n_shapley_values_neg = {
        ...     1: np.asarray([0, -1.5, 0]),
        ...     2: np.asarray([-0.25, -0.5, -0.75]),
        ...     3: np.asarray([-0.5, -0.25, -0.25]),
        ... }
        >>> feature_names = ["a", "b", "c"]
        >>> fig, axes = stacked_bar_plot(
        ...     feature_names=feature_names,
        ...     n_shapley_values_pos=n_shapley_values_pos,
        ...     n_shapley_values_neg=n_shapley_values_neg,
        ... )
        >>> plt.show()
    """
    # sanitize inputs
    if n_sii_max_order is None:
        n_sii_max_order = len(n_shapley_values_pos)

    fig, axis = plt.subplots()

    # transform data to make plotting easier
    n_features = len(feature_names)
    x = np.arange(n_features)
    values_pos = np.array(
        [values for order, values in n_shapley_values_pos.items() if order >= n_sii_max_order]
    )
    values_neg = np.array(
        [values for order, values in n_shapley_values_neg.items() if order >= n_sii_max_order]
    )

    # get helper variables for plotting the bars
    min_max_values = [0, 0]  # to set the y-axis limits after all bars are plotted
    reference_pos = np.zeros(n_features)  # to plot the bars on top of each other
    reference_neg = deepcopy(values_neg[0])  # to plot the bars below of each other

    # plot the bar segments
    for order in range(len(values_pos)):
        axis.bar(x, height=values_pos[order], bottom=reference_pos, color=COLORS_N_SII[order])
        axis.bar(x, height=abs(values_neg[order]), bottom=reference_neg, color=COLORS_N_SII[order])
        axis.axhline(y=0, color="black", linestyle="solid", linewidth=0.5)
        reference_pos += values_pos[order]
        try:
            reference_neg += values_neg[order + 1]
        except IndexError:
            pass
        min_max_values[0] = min(min_max_values[0], min(reference_neg))
        min_max_values[1] = max(min_max_values[1], max(reference_pos))

    # add a legend to the plots
    legend_elements = []
    for order in range(n_sii_max_order):
        legend_elements.append(
            Patch(facecolor=COLORS_N_SII[order], edgecolor="black", label=f"Order {order + 1}")
        )
    axis.legend(handles=legend_elements, loc="upper center", ncol=min(n_sii_max_order, 4))

    x_ticks_labels = [feature for feature in feature_names]  # might be unnecessary
    axis.set_xticks(x)
    axis.set_xticklabels(x_ticks_labels, rotation=45, ha="right")

    axis.set_xlim(-0.5, n_features - 0.5)
    axis.set_ylim(
        min_max_values[0] - abs(min_max_values[1] - min_max_values[0]) * 0.02,
        min_max_values[1] + abs(min_max_values[1] - min_max_values[0]) * 0.3,
    )

    # set title and labels if not provided

    (
        axis.set_title(f"n-SII values up to order ${n_sii_max_order}$")
        if title is None
        else axis.set_title(title)
    )

    axis.set_xlabel("features") if xlabel is None else axis.set_xlabel(xlabel)
    axis.set_ylabel("n-SII values") if ylabel is None else axis.set_ylabel(ylabel)

    plt.tight_layout()

    return fig, axis
