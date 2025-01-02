"""Wrapper for the bar plot from the ``shap`` package."""

import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..interaction_values import InteractionValues
from ._config import BLUE, RED
from .utils import get_interaction_values_and_feature_names

__all__ = ["bar_plot"]


def format_value(s, format_str):
    """Strips trailing zeros and uses a unicode minus sign."""
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r"\.?0+$", "", s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return s


def _bar(values, feature_names, max_display=10, ax=None, show=True):
    """Create a bar plot of a set of SHAP values.

    Parameters
    ----------
    shap_values : shap.Explanation or shap.Cohorts or dictionary of shap.Explanation objects
        Passing a multi-row :class:`.Explanation` object creates a global
        feature importance plot.

        Passing a single row of an explanation (i.e. ``shap_values[0]``) creates
        a local feature importance plot.

        Passing a dictionary of Explanation objects will create a multiple-bar
        plot with one bar type for each of the cohorts represented by the
        explanation objects.
    max_display : int
        How many top features to include in the bar plot (default is 10).
    order : OpChain or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values
        and an axis, or a direct sample ordering given as a ``numpy.ndarray``.

        By default, take the absolute value.
    clustering: np.ndarray or None
        A partition tree, as returned by ``shap.utils.hclust``
    clustering_cutoff: float
        Controls how much of the clustering structure is displayed.
    show_data: bool or str
        Controls if data values are shown as part of the y tick labels. If
        "auto", we show the data only when there are no transforms.
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.
    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot drawn onto it. Only returned if ``show=False``.

    Examples
    --------
    See `bar plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html>`_.

    """
    # assert str(type(shap_values)).endswith("Explanation'>"), "The shap_values parameter must be a shap.Explanation object!"

    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([f"Feature {i}" for i in range(len(values[0]))])
    if issubclass(type(feature_names), str):
        feature_names = [i + " " + feature_names for i in range(len(values[0]))]

    # build our auto xlabel based on the transform history of the Explanation object
    xlabel = "SHAP value"

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values[0]))
    max_display = min(max_display, num_features)

    # Make it descending order
    feature_order = np.argsort(values)[0][::-1]

    y_pos = np.arange(len(feature_order), 0, -1)

    # build our y-tick labels
    yticklabels = []
    for i in feature_order:
        yticklabels.append(feature_names[i])

    if ax is None:
        ax = plt.gca()
        # Only modify the figure size if ax was not passed in
        # compute our figure size based on how many features we are showing
        fig = plt.gcf()
        row_height = 0.5
        fig.set_size_inches(8, num_features * row_height * np.sqrt(len(values)) + 1.5)

    # if negative values are present then we draw a vertical line to mark 0, otherwise the axis does this for us...
    negative_values_present = np.sum(values[:, feature_order[:num_features]] < 0) > 0
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
            values[i, feature_order],
            bar_width,
            align="center",
            color=[
                BLUE.hex if values[i, feature_order[j]] <= 0 else RED.hex for j in range(len(y_pos))
            ],
            hatch=patterns[i],
            edgecolor=(1, 1, 1, 0.8),
            label="",
        )

    # draw the yticks (the 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks)
    ax.set_yticks(
        list(y_pos) + list(y_pos + 1e-8),
        yticklabels + [t.split("=")[-1] for t in yticklabels],
        fontsize=13,
    )

    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    # xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width

    for i in range(len(values)):
        ypos_offset = -((i - len(values) / 2) * bar_width + bar_width / 2)
        for j in range(len(y_pos)):
            ind = feature_order[j]
            if values[i, ind] < 0:
                ax.text(
                    values[i, ind] - (5 / 72) * bbox_to_xscale,
                    y_pos[j] + ypos_offset,
                    format_value(values[i, ind], "%+0.02f"),
                    horizontalalignment="right",
                    verticalalignment="center",
                    color=BLUE.hex,
                    fontsize=12,
                )
            else:
                ax.text(
                    values[i, ind] + (5 / 72) * bbox_to_xscale,
                    y_pos[j] + ypos_offset,
                    format_value(values[i, ind], "%+0.02f"),
                    horizontalalignment="left",
                    verticalalignment="center",
                    color=RED.hex,
                    fontsize=12,
                )

    # put horizontal lines for each feature row
    for i in range(num_features):
        ax.axhline(i + 1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if negative_values_present:
        ax.spines["left"].set_visible(False)
    ax.tick_params("x", labelsize=11)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_buffer = (xmax - xmin) * 0.05

    if negative_values_present:
        ax.set_xlim(xmin - x_buffer, xmax + x_buffer)
    else:
        ax.set_xlim(xmin, xmax + x_buffer)

    # if features is None:
    #     pl.xlabel(labels["GLOBAL_VALUE"], fontsize=13)
    # else:
    ax.set_xlabel(xlabel, fontsize=13)

    if len(values) > 1:
        ax.legend(fontsize=12)

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    if show:
        plt.show()
    else:
        return ax


def bar_plot(
    list_of_interaction_values: list[InteractionValues],
    feature_names: Optional[np.ndarray] = None,
    show: bool = False,
    abbreviate: bool = True,
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
        abbreviate: Whether to abbreviate the feature names. Defaults to ``True``.
        **kwargs: Keyword arguments passed to ``shap.plots.beeswarm()``.
    """

    assert len(np.unique([iv.max_order for iv in list_of_interaction_values])) == 1

    _global_values = []
    _base_values = []
    _labels = []
    _first_iv = True
    for iv in list_of_interaction_values:

        _shap_values, _names = get_interaction_values_and_feature_names(
            iv, feature_names, None, abbreviate=abbreviate
        )
        if _first_iv:
            _labels = _names
            _first_iv = False
        _global_values.append(_shap_values)
        _base_values.append(iv.baseline_value)

    _labels = np.array(_labels) if feature_names is not None else None

    ax = _bar(values=np.stack(_global_values), feature_names=_labels, show=False)
    ax.set_xlabel("mean(|Shapley Interaction value|)")
    if not show:
        return ax
    plt.show()
