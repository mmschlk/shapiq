"""Wrapper for the bar plot from the ``shap`` package.

Note:
    Code and implementation was taken and adapted from the [SHAP package](https://github.com/shap/shap)
    which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues, aggregate_interaction_values

from ._config import BLUE, RED
from .utils import abbreviate_feature_names, format_labels, format_value

__all__ = ["bar_plot"]


def _bar(
    values: np.ndarray,
    feature_names: np.ndarray,
    max_display: int | None = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Create a bar plot of a set of SHAP values.

    Note:
        This function was taken and adapted from the [SHAP package](https://github.com/shap/shap/blob/master/shap/plots/_bar.py)
        which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).
        Do not use this function directly, use the ``bar_plot`` function instead.

    Args:
        values: The explanation values to plot as a 2D array. Each row should be a different group
            of values to plot. The columns are the feature values.
        feature_names: The names of the features to display.
        max_display: The maximum number of features to display. Defaults to ``10``.
        ax: The axis to plot on. If ``None``, a new figure and axis is created. Defaults to
            ``None``.

    Returns:
        The axis of the plot.

    """
    # determine how many top features we will plot
    num_features = len(values[0])
    if max_display is None:
        max_display = num_features
    max_display = min(max_display, num_features)
    num_cut = max(num_features - max_display, 0)  # number of features that are not displayed

    # get order of features in descending order
    feature_order = np.argsort(np.mean(values, axis=0))[::-1]

    # if there are more features than we are displaying then we aggregate the features not shown
    if num_cut > 0:
        cut_feature_values = values[:, feature_order[max_display:]]
        sum_of_remaining = np.sum(cut_feature_values, axis=None)
        index_of_last = feature_order[max_display]
        values[:, index_of_last] = sum_of_remaining
        max_display += 1  # include the sum of the remaining in the display

    # get the top features and their names
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    yticklabels = [feature_names[i] for i in feature_inds]
    if num_cut > 0:
        yticklabels[-1] = f"Sum of {int(num_cut)} other features"

    # create a figure if one was not provided
    if ax is None:
        ax = plt.gca()
        # only modify the figure size if ax was not passed in
        # compute our figure size based on how many features we are showing
        fig = plt.gcf()
        row_height = 0.5
        fig.set_size_inches(
            8 + 0.3 * max([len(feature_name) for feature_name in feature_names]),
            max_display * row_height * np.sqrt(len(values)) + 1.5,
        )

    # if negative values are present, we draw a vertical line to mark 0
    negative_values_present = np.sum(values[:, feature_order[:max_display]] < 0) > 0
    if negative_values_present:
        ax.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    # draw the bars
    patterns = (None, "\\\\", "++", "xx", "////", "*", "o", "O", ".", "-")
    total_width = 0.7
    bar_width = total_width / len(values)
    for i in range(len(values)):
        ypos_offset = -((i - len(values) / 2) * bar_width + bar_width / 2)
        ax.barh(
            y_pos + ypos_offset,
            values[i, feature_inds],
            bar_width,
            align="center",
            color=[
                BLUE.hex if values[i, feature_inds[j]] <= 0 else RED.hex for j in range(len(y_pos))
            ],
            hatch=patterns[i],
            edgecolor=(1, 1, 1, 0.8),
            label="Group " + str(i + 1),
        )

    # draw the yticks (the 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks)
    ax.set_yticks(
        list(y_pos) + list(y_pos + 1e-8),
        yticklabels + [t.split("=")[-1] for t in yticklabels],
        fontsize=13,
    )

    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width

    # draw the bar labels as text next to the bars
    for i in range(len(values)):
        ypos_offset = -((i - len(values) / 2) * bar_width + bar_width / 2)
        for j in range(len(y_pos)):
            ind = feature_inds[j]
            if values[i, ind] < 0:
                ax.text(
                    values[i, ind] - (5 / 72) * bbox_to_xscale,
                    float(y_pos[j] + ypos_offset),
                    format_value(values[i, ind], "%+0.02f"),
                    horizontalalignment="right",
                    verticalalignment="center",
                    color=BLUE.hex,
                    fontsize=12,
                )
            else:
                ax.text(
                    values[i, ind] + (5 / 72) * bbox_to_xscale,
                    float(y_pos[j] + ypos_offset),
                    format_value(values[i, ind], "%+0.02f"),
                    horizontalalignment="left",
                    verticalalignment="center",
                    color=RED.hex,
                    fontsize=12,
                )

    # put horizontal lines for each feature row
    for i in range(max_display):
        ax.axhline(i + 1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)

    # remove plot frame and y-axis ticks
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if negative_values_present:
        ax.spines["left"].set_visible(False)
    ax.tick_params("x", labelsize=11)

    # set the x-axis limits to cover the data
    xmin, xmax = ax.get_xlim()
    x_buffer = (xmax - xmin) * 0.05
    if negative_values_present:
        ax.set_xlim(xmin - x_buffer, xmax + x_buffer)
    else:
        ax.set_xlim(xmin, xmax + x_buffer)

    ax.set_xlabel("Attribution", fontsize=13)

    if len(values) > 1:
        ax.legend(fontsize=12, loc="lower right")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(max_display):
        tick_labels[i].set_color("#999999")

    return ax


def bar_plot(
    list_of_interaction_values: list[InteractionValues],
    *,
    feature_names: np.ndarray | None = None,
    show: bool = False,
    abbreviate: bool = True,
    max_display: int | None = 10,
    global_plot: bool = True,
    plot_base_value: bool = False,
) -> plt.Axes | None:
    """Draws interaction values as a SHAP bar plot[1]_.

    The function draws the interaction values on a bar plot. The interaction values can be
    aggregated into a global explanation or plotted separately.

    Args:
        list_of_interaction_values: A list containing InteractionValues objects.
        feature_names: The feature names used for plotting. If no feature names are provided, the
            feature indices are used instead. Defaults to ``None``.
        show: Whether ``matplotlib.pyplot.show()`` is called before returning. Default is ``True``.
            Setting this to ``False`` allows the plot to be customized further after it has been
            created.
        abbreviate: Whether to abbreviate the feature names. Defaults to ``True``.
        max_display: The maximum number of features to display. Defaults to ``10``. If set to
            ``None``, all features are displayed.
        global_plot: Weather to aggregate the values of the different InteractionValues objects
            into a global explanation (``True``) or to plot them as separate bars (``False``).
            Defaults to ``True``. If only one InteractionValues object is provided, this parameter
            is ignored.
        plot_base_value: Whether to include the base value in the plot or not. Defaults to
            ``False``.

    Returns:
        If ``show`` is ``False``, the function returns the axis of the plot. Otherwise, it returns
        ``None``.

    References:
        .. [1] SHAP is available at https://github.com/shap/shap

    """
    n_players = list_of_interaction_values[0].n_players

    if feature_names is not None:
        if abbreviate:
            feature_names = abbreviate_feature_names(feature_names)
        feature_mapping = {i: feature_names[i] for i in range(n_players)}
    else:
        feature_mapping = {i: "F" + str(i) for i in range(n_players)}

    # aggregate the interaction values if global_plot is True
    if global_plot and len(list_of_interaction_values) > 1:
        # The aggregation of the global values will be done on the absolute values
        list_of_interaction_values = [abs(iv) for iv in list_of_interaction_values]
        global_values = aggregate_interaction_values(list_of_interaction_values, aggregation="mean")
        values = np.expand_dims(global_values.values, axis=0)
        interaction_list = global_values.interaction_lookup.keys()
    else:  # plot the interaction values separately  (also includes the case of a single object)
        all_interactions = set()
        for iv in list_of_interaction_values:
            all_interactions.update(iv.interaction_lookup.keys())
        all_interactions = sorted(all_interactions)
        interaction_list = []
        values = np.zeros((len(list_of_interaction_values), len(all_interactions)))
        for j, interaction in enumerate(all_interactions):
            interaction_list.append(interaction)
            for i, iv in enumerate(list_of_interaction_values):
                values[i, j] = iv[interaction]

    # Include the base value in the plot
    if not plot_base_value:
        values = values[:, 1:]
        interaction_list = list(interaction_list)[1:]

    # format the labels
    labels = [format_labels(feature_mapping, interaction) for interaction in interaction_list]

    ax = _bar(values=values, feature_names=labels, max_display=max_display)
    if not show:
        return ax
    plt.show()
    return None
