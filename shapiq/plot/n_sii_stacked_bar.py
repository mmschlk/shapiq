"""This module contains functions to plot the n_sii stacked bar charts."""
__all__ = ["n_sii_stacked_bar_plot"]

from copy import deepcopy
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from _config import COLORS_N_SII


def n_sii_stacked_bar_plot(
    feature_names: Union[list, np.ndarray],
    n_shapley_values_pos: dict,
    n_shapley_values_neg: dict,
    n_sii_order: int,
):
    """Plot the n-SII values for a given instance.

    Args:
        feature_names (list): The names of the features.
        n_shapley_values_pos (dict): The positive n-SII values.
        n_shapley_values_neg (dict): The negative n-SII values.
        n_sii_order (int): The order of the n-SII values.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing the figure and
            the axis of the plot.
    """
    fig, axis = plt.subplots(figsize=(6, 4.15))

    # transform data to make plotting easier
    n_features = len(feature_names)
    x = np.arange(n_features)
    values_pos = np.array([values for order, values in n_shapley_values_pos.items()])
    values_neg = np.array([values for order, values in n_shapley_values_neg.items()])

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
        except KeyError:
            pass
        min_max_values[0] = min(min_max_values[0], min(reference_neg))
        min_max_values[1] = max(min_max_values[1], max(reference_pos))

    # add a legend to the plots
    legend_elements = []
    for order in range(n_sii_order):
        legend_elements.append(
            Patch(facecolor=COLORS_N_SII[order], edgecolor="black", label=f"Order {order + 1}")
        )
    axis.legend(handles=legend_elements, loc="upper center", ncol=min(n_sii_order, 4))

    x_ticks_labels = [feature for feature in feature_names]  # might be unnecessary
    axis.set_xticks(x)
    axis.set_xticklabels(x_ticks_labels, rotation=45, ha="right")

    axis.set_xlim(-0.5, n_features - 0.5)
    axis.set_ylim(
        min_max_values[0] - abs(min_max_values[1] - min_max_values[0]) * 0.02,
        min_max_values[1] + abs(min_max_values[1] - min_max_values[0]) * 0.3,
    )

    axis.set_ylabel("n-SII values")
    axis.set_xlabel("features")
    axis.set_title(f"n-SII values up to order ${n_sii_order}$")

    plt.tight_layout()

    return fig, axis
