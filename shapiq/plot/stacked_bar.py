"""This module contains functions to plot the n_sii stacked bar charts."""

from __future__ import annotations

import contextlib
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ._config import COLORS_K_SII

__all__ = ["stacked_bar_plot"]


if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


def stacked_bar_plot(
    interaction_values: InteractionValues,
    *,
    feature_names: list[Any] | None = None,
    max_order: int | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes] | None:
    """The stacked bar plot interaction scores.

    This stacked bar plot can be used to visualize the amount of interaction between the features
    for a given instance. The interaction values are plotted as stacked bars with positive and
    negative parts stacked on top of each other. The colors represent the order of the
    interaction values. For a detailed explanation of this plot, we refer to Bordt and von Luxburg
    (2023)[1]_.

    An example of the plot is shown below.

    .. image:: /_static/stacked_bar_exampl.png
        :width: 400
        :align: center

    Args:
        interaction_values(InteractionValues): n-SII values as InteractionValues object
        feature_names: The feature names used for plotting. If no feature names are provided, the
            feature indices are used instead. Defaults to ``None``.
        max_order (int): The order of the n-SII values.
        title (str): The title of the plot.
        xlabel (str): The label of the x-axis.
        ylabel (str): The label of the y-axis.
        show (bool): Whether to show the plot. Defaults to ``False``.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing the figure and
            the axis of the plot.

    Note:
        To change the figure size, font size, etc., use the [matplotlib parameters](https://matplotlib.org/stable/users/explain/customizing.html).

    Example:
        >>> import numpy as np
        >>> from shapiq.plot import stacked_bar_plot
        >>> interaction_values = InteractionValues(
        ...    values=np.array([1, -1.5, 1.75, 0.25, -0.5, 0.75,0.2]),
        ...    index="SII",
        ...    min_order=1,
        ...    max_order=3,
        ...    n_players=3,
        ...    baseline_value=0
        ... )
        >>> feature_names = ["a", "b", "c"]
        >>> fig, axes = stacked_bar_plot(
        ...     interaction_values=interaction_values,
        ...     feature_names=feature_names,
        ... )
        >>> plt.show()

    References:
        .. [1] Bordt, M., and von Luxburg, U. (2023). From Shapley Values to Generalized Additive Models and back. Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, PMLR 206:709-745. url: https://proceedings.mlr.press/v206/bordt23a.html

    """
    # sanitize inputs
    if max_order is None:
        max_order = interaction_values.max_order

    fig, axis = plt.subplots()

    # transform data to make plotting easier
    values_pos = np.array(
        [
            interaction_values.get_n_order_values(order)
            .clip(min=0)
            .sum(axis=tuple(range(1, order)))
            for order in range(1, max_order + 1)
        ],
    )
    values_neg = np.array(
        [
            interaction_values.get_n_order_values(order)
            .clip(max=0)
            .sum(axis=tuple(range(1, order)))
            for order in range(1, max_order + 1)
        ],
    )
    # get the number of features and the feature names
    n_features = len(values_pos[0])
    if feature_names is None:
        feature_names = [str(i + 1) for i in range(n_features)]
    x = np.arange(n_features)

    # get helper variables for plotting the bars
    min_max_values = [0, 0]  # to set the y-axis limits after all bars are plotted
    reference_pos = np.zeros(n_features)  # to plot the bars on top of each other
    reference_neg = deepcopy(values_neg[0])  # to plot the bars below of each other

    # plot the bar segments
    for order in range(len(values_pos)):
        axis.bar(x, height=values_pos[order], bottom=reference_pos, color=COLORS_K_SII[order])
        axis.bar(x, height=abs(values_neg[order]), bottom=reference_neg, color=COLORS_K_SII[order])
        axis.axhline(y=0, color="black", linestyle="solid", linewidth=0.5)
        reference_pos += values_pos[order]
        with contextlib.suppress(IndexError):
            reference_neg += values_neg[order + 1]
        min_max_values[0] = min(min_max_values[0], *reference_neg)
        min_max_values[1] = max(min_max_values[1], *reference_pos)

    # add a legend to the plots
    legend_elements = [
        Patch(facecolor=COLORS_K_SII[order], edgecolor="black", label=f"Order {order + 1}")
        for order in range(max_order)
    ]
    axis.legend(handles=legend_elements, loc="upper center", ncol=min(max_order, 4))

    x_ticks_labels = list(feature_names)  # might be unnecessary
    axis.set_xticks(x)
    axis.set_xticklabels(x_ticks_labels, rotation=45, ha="right")

    axis.set_xlim(-0.5, n_features - 0.5)
    axis.set_ylim(
        min_max_values[0] - abs(min_max_values[1] - min_max_values[0]) * 0.02,
        min_max_values[1] + abs(min_max_values[1] - min_max_values[0]) * 0.3,
    )

    # set title and labels if not provided
    if title is not None:
        axis.set_title(title)

    axis.set_xlabel("features") if xlabel is None else axis.set_xlabel(xlabel)
    axis.set_ylabel("SI values") if ylabel is None else axis.set_ylabel(ylabel)

    plt.tight_layout()

    if not show:
        return fig, axis
    plt.show()
    return None
