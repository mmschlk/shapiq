"""Wrapper for the waterfall plot from the ``shap`` package.

Note:
    Code and implementation was taken and adapted from the [SHAP package](https://github.com/shap/shap)
    which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ._config import BLUE, RED
from .utils import abbreviate_feature_names, format_labels, format_value

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

__all__ = ["waterfall_plot"]


def _draw_waterfall_plot(
    values: np.ndarray,
    base_values: float,
    feature_names: list[str],
    *,
    max_display: int = 10,
    show: bool = False,
) -> plt.Axes | None:
    """The waterfall plot from the SHAP package.

    Note:
        This function was taken and adapted from the [SHAP package](https://github.com/shap/shap/blob/master/shap/plots/_waterfall.py)
        which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).
        Do not use this function directly, use the ``waterfall_plot`` function instead.

    Args:
        values: The values to plot.
        base_values: The base value.
        feature_names: The names of the features.
        max_display: The maximum number of features to display.
        show: Whether to show the plot.

    Returns:
        The plot if ``show`` is ``False``.

    """
    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 3.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    num_individual = num_features if num_features == len(values) else num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color="#bbbbbb",
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )
        yticklabels[rng[i]] = feature_names[order[i]]

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = f"{int(len(values) - num_features + 1)} other features"
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=RED.hex,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=BLUE.hex,
        alpha=0,
    )

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    dpi = fig.dpi
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i],
            pos_inds[i],
            max(dist - hl_scaled, 0.000001),
            0,
            head_length=min(dist, hl_scaled),
            color=RED.hex,
            width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=BLUE.hex,
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5 * dist,
            pos_inds[i],
            format_value(pos_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                pos_inds[i],
                format_value(pos_widths[i], "%+0.02f"),
                horizontalalignment="left",
                verticalalignment="center",
                color=RED.hex,
                fontsize=12,
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i],
            neg_inds[i],
            -max(-dist - hl_scaled, 0.000001),
            0,
            head_length=min(-dist, hl_scaled),
            color=BLUE.hex,
            width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=RED.hex,
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5 * dist,
            neg_inds[i],
            format_value(neg_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            plt.text(
                neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                neg_inds[i],
                format_value(neg_widths[i], "%+0.02f"),
                horizontalalignment="right",
                verticalalignment="center",
                color=BLUE.hex,
                fontsize=12,
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)
    plt.yticks(
        ytick_pos,
        yticklabels[:-1] + [label.split("=")[-1] for label in yticklabels[:-1]],
        fontsize=13,
    )

    # Check that the y-ticks are not drawn outside the plot
    max_label_width = (
        max([label.get_window_extent(renderer=renderer).width for label in ax.get_yticklabels()])
        / dpi
    )
    if max_label_width > 0.1 * fig.get_size_inches()[0]:
        required_width = max_label_width / 0.1
        fig_height = fig.get_size_inches()[1]
        fig.set_size_inches(required_width, fig_height, forward=True)

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(
        base_values,
        0,
        1 / num_features,
        color="#bbbbbb",
        linestyle="--",
        linewidth=0.5,
        zorder=-1,
    )
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks(
        [base_values, base_values + 1e-8],
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(
        ["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"],
        fontsize=12,
        ha="left",
    )
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8])
    ax3.set_xticklabels(
        ["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"],
        fontsize=12,
        ha="left",
    )
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + mpl.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans),
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + mpl.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans),
    )
    tick_labels[1].set_color("#999999")
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + mpl.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans),
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + mpl.transforms.ScaledTranslation(22 / 72.0, -1 / 72.0, fig.dpi_scale_trans),
    )

    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    if show:
        plt.show()
        return None
    return plt.gca()


def waterfall_plot(
    interaction_values: InteractionValues,
    *,
    feature_names: np.ndarray[str] | None = None,
    show: bool = False,
    max_display: int = 10,
    abbreviate: bool = True,
) -> plt.Axes | None:
    """Draws a waterfall plot with the interaction values.

    The waterfall plot shows the individual contributions of the features to the interaction values.
    The plot is based on the waterfall plot from the SHAP[1]_ package.

    Args:
        interaction_values: The interaction values as an interaction object.
        feature_names: The names of the features. Defaults to ``None``.
        show: Whether to show the plot. Defaults to ``False``.
        max_display: The maximum number of interactions to display. Defaults to ``10``.
        abbreviate: Whether to abbreviate the feature names. Defaults to ``True``.

    Returns:
        The plot if ``show`` is ``False``.

    References:
        .. [1] SHAP is available at https://github.com/shap/shap

    """
    if feature_names is None:
        feature_mapping = {i: str(i) for i in range(interaction_values.n_players)}
    else:
        if abbreviate:
            feature_names = abbreviate_feature_names(feature_names)
        feature_mapping = {i: feature_names[i] for i in range(interaction_values.n_players)}

    # create the data for the waterfall plot in the correct format
    data = []
    for feature_tuple, value in interaction_values.dict_values.items():
        if len(feature_tuple) > 0:
            data.append((format_labels(feature_mapping, feature_tuple), str(value)))
    data = np.array(data, dtype=object)
    values = data[:, 1].astype(float)
    feature_names = data[:, 0]

    return _draw_waterfall_plot(
        values,
        interaction_values.baseline_value,
        feature_names,
        max_display=max_display,
        show=show,
    )
