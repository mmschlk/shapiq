"""Wrapper for the force plot from the ``shap`` package.

Note:
    Code and implementation was taken and adapted from the [SHAP package](https://github.com/shap/shap)
    which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from .utils import abbreviate_feature_names, format_labels

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

__all__ = ["force_plot"]


def _create_bars(
    out_value: float,
    features: np.ndarray,
    feature_type: str,
    width_separators: float,
    width_bar: float,
) -> tuple[list, list]:
    rectangle_list = []
    separator_list = []

    pre_val = out_value
    for index, feature_iteration in zip(range(len(features)), features, strict=False):
        if feature_type == "positive":
            left_bound = float(feature_iteration[0])
            right_bound = pre_val
            pre_val = left_bound

            separator_indent = np.abs(width_separators)
            separator_pos = left_bound
            colors = ["#FF0D57", "#FFC3D5"]
        else:
            left_bound = pre_val
            right_bound = float(feature_iteration[0])
            pre_val = right_bound

            separator_indent = -np.abs(width_separators)
            separator_pos = right_bound
            colors = ["#1E88E5", "#D1E6FA"]

        # Create rectangle
        if index == 0:
            if feature_type == "positive":
                points_rectangle = [
                    [left_bound, 0],
                    [right_bound, 0],
                    [right_bound, width_bar],
                    [left_bound, width_bar],
                    [left_bound + separator_indent, (width_bar / 2)],
                ]
            else:
                points_rectangle = [
                    [right_bound, 0],
                    [left_bound, 0],
                    [left_bound, width_bar],
                    [right_bound, width_bar],
                    [right_bound + separator_indent, (width_bar / 2)],
                ]

        else:
            points_rectangle = [
                [left_bound, 0],
                [right_bound, 0],
                [right_bound + separator_indent * 0.90, (width_bar / 2)],
                [right_bound, width_bar],
                [left_bound, width_bar],
                [left_bound + separator_indent * 0.90, (width_bar / 2)],
            ]

        line = plt.Polygon(
            points_rectangle,
            closed=True,
            fill=True,
            facecolor=colors[0],
            linewidth=0,
        )
        rectangle_list += [line]

        # Create separator
        points_separator = [
            [separator_pos, 0],
            [separator_pos + separator_indent, (width_bar / 2)],
            [separator_pos, width_bar],
        ]

        line = plt.Polygon(points_separator, closed=None, fill=None, edgecolor=colors[1], lw=3)
        separator_list += [line]

    return rectangle_list, separator_list


def _add_labels(
    fig: plt.Figure,
    ax: plt.Axes,
    out_value: float,
    features: np.ndarray,
    feature_type: str,
    offset_text: float,
    total_effect: float = 0,
    min_perc: float = 0.05,
    text_rotation: float = 0,
) -> None:
    """Add labels to the plot.

    Args:
        fig: Figure of the plot
        ax: Axes of the plot
        out_value: output value
        features: The values and names of the features
        feature_type: Indicating whether positive or negative features
        offset_text: value to offset name of the features
        total_effect: Total value of all features. Used to filter out features that do not contribute at least min_perc to the total effect.
        Defaults to 0 indicating that all features are shown.
        min_perc: minimal percentage of the total effect that a feature must contribute to be shown. Defaults to 0.05.
        text_rotation: Degree the text should be rotated. Defaults to 0.
    """
    start_text = out_value
    pre_val = out_value

    # Define variables specific to positive and negative effect features
    if feature_type == "positive":
        colors = ["#FF0D57", "#FFC3D5"]
        alignment = "right"
        sign = 1
    else:
        colors = ["#1E88E5", "#D1E6FA"]
        alignment = "left"
        sign = -1

    # Draw initial line
    if feature_type == "positive":
        x, y = np.array([[pre_val, pre_val], [0, -0.18]])
        line = lines.Line2D(x, y, lw=1.0, alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val

    box_end = out_value
    val = out_value
    for feature in features:
        # Exclude all labels that do not contribute at least 10% to the total
        feature_contribution = np.abs(float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break

        # Compute value for current feature
        val = float(feature[0])

        # Draw labels.
        text = feature[1]

        va_alignment = "top" if text_rotation != 0 else "baseline"

        text_out_val = plt.text(
            start_text - sign * offset_text,
            -0.15,
            text,
            fontsize=12,
            color=colors[0],
            horizontalalignment=alignment,
            va=va_alignment,
            rotation=text_rotation,
        )
        text_out_val.set_bbox({"facecolor": "none", "edgecolor": "none"})

        # We need to draw the plot to be able to get the size of the
        # text box
        fig.canvas.draw()
        box_size = text_out_val.get_bbox_patch().get_extents().transformed(ax.transData.inverted())
        if feature_type == "positive":
            box_end_ = box_size.get_points()[0][0]
        else:
            box_end_ = box_size.get_points()[1][0]

        # Create end line
        if (sign * box_end_) > (sign * val):
            x, y = np.array([[val, val], [0, -0.18]])
            line = lines.Line2D(x, y, lw=1.0, alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val

        else:
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[val, box_end, box_end], [0, -0.08, -0.18]])
            line = lines.Line2D(x, y, lw=1.0, alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end

        # Update previous value
        pre_val = float(feature[0])

    # Create line for labels
    extent_shading = [out_value, box_end, 0, -0.31]
    path = [
        [out_value, 0],
        [pre_val, 0],
        [box_end, -0.08],
        [box_end, -0.2],
        [out_value, -0.2],
        [out_value, 0],
    ]

    path = Path(path)
    patch = PathPatch(path, facecolor="none", edgecolor="none")
    ax.add_patch(patch)

    # Extend axis if needed
    lower_lim, upper_lim = ax.get_xlim()
    if box_end < lower_lim:
        ax.set_xlim(box_end, upper_lim)

    if box_end > upper_lim:
        ax.set_xlim(lower_lim, box_end)

    # Create shading
    if feature_type == "positive":
        colors = np.array([(255, 13, 87), (255, 255, 255)]) / 255.0
    else:
        colors = np.array([(30, 136, 229), (255, 255, 255)]) / 255.0

    cm = mpl.colors.LinearSegmentedColormap.from_list("cm", colors)

    _, z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
    im = plt.imshow(
        z2,
        interpolation="quadric",
        cmap=cm,
        vmax=0.01,
        alpha=0.3,
        origin="lower",
        extent=extent_shading,
        clip_path=patch,
        clip_on=True,
        aspect="auto",
    )
    im.set_clip_path(patch)

    return fig, ax


def _add_output_element(out_name: str, out_value: float, ax: plt.Axes) -> None:
    """Add grew line indicating the output value to the plot.

    Args:
        out_name: Name of the output value
        out_value: Value of the output
        ax: Axis of the plot

    Returns: Nothing

    """
    # Add output value
    x, y = np.array([[out_value, out_value], [0, 0.24]])
    line = lines.Line2D(x, y, lw=2.0, color="#F2F2F2")
    line.set_clip_on(False)
    ax.add_line(line)

    font0 = FontProperties()
    font = font0.copy()
    font.set_weight("bold")
    text_out_val = plt.text(
        out_value,
        0.25,
        f"{out_value:.2f}",
        fontproperties=font,
        fontsize=14,
        horizontalalignment="center",
    )
    text_out_val.set_bbox({"facecolor": "white", "edgecolor": "white"})

    text_out_val = plt.text(
        out_value,
        0.33,
        out_name,
        fontsize=12,
        alpha=0.5,
        horizontalalignment="center",
    )
    text_out_val.set_bbox({"facecolor": "white", "edgecolor": "white"})


def _add_base_value(base_value: float, ax: plt.Axes) -> None:
    """Add base value to the plot.

    Args:
        base_value: the base value of the game
        ax: Axes of the plot

    Returns: None

    """
    x, y = np.array([[base_value, base_value], [0.13, 0.25]])
    line = lines.Line2D(x, y, lw=2.0, color="#F2F2F2")
    line.set_clip_on(False)
    ax.add_line(line)

    text_out_val = ax.text(
        base_value,
        0.25,
        "base value",
        fontsize=12,
        alpha=1,
        horizontalalignment="center",
    )
    text_out_val.set_bbox({"facecolor": "white", "edgecolor": "white"})


def update_axis_limits(
    ax: plt.Axes,
    total_pos: float,
    pos_features: np.ndarray,
    total_neg: float,
    neg_features: np.ndarray,
    base_value: float,
    out_value: float,
) -> None:
    """Adjust the axis limits of the plot according to values.

    Args:
        ax: Axes of the plot
        total_pos: value of the total positive features
        pos_features: values and names of the positive features
        total_neg: value of the total negative features
        neg_features: values and names of the negative features
        base_value: the base value of the game
        out_value: the output value

    Returns: None

    """
    ax.set_ylim(-0.5, 0.15)
    padding = np.max([np.abs(total_pos) * 0.2, np.abs(total_neg) * 0.2])

    if len(pos_features) > 0:
        min_x = min(np.min(pos_features[:, 0].astype(float)), base_value) - padding
    else:
        min_x = out_value - padding
    if len(neg_features) > 0:
        max_x = max(np.max(neg_features[:, 0].astype(float)), base_value) + padding
    else:
        max_x = out_value + padding
    ax.set_xlim(min_x, max_x)

    plt.tick_params(
        top=True,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labeltop=True,
        labelbottom=False,
    )
    plt.locator_params(axis="x", nbins=12)

    for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values(), strict=False):
        if key != "top":
            spine.set_visible(False)


def _split_features(
    interaction_dictionary: dict[tuple[int, ...], float],
    feature_to_names: dict[int, str],
    out_value: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Splits the features into positive and negative values.

    Args:
        interaction_dictionary: Dictionary containing the interaction values mapping from
            feature indices to their values.
        feature_to_names: Dictionary mapping feature indices to feature names.
        out_value: The output value.

    Returns:
        tuple: A tuple containing the positive features, negative features, total positive value,
            and total negative value.

    """
    # split features into positive and negative values
    pos_features, neg_features = [], []
    for coaltion, value in interaction_dictionary.items():
        if len(coaltion) == 0:
            continue
        label = format_labels(feature_to_names, coaltion)
        if value >= 0:
            pos_features.append([str(value), label])
        elif value < 0:
            neg_features.append([str(value), label])
    # sort feature values descending according to (absolute) features values
    pos_features = sorted(pos_features, key=lambda x: float(x[0]), reverse=True)
    neg_features = sorted(neg_features, key=lambda x: float(x[0]), reverse=False)
    pos_features = np.array(pos_features, dtype=object)
    neg_features = np.array(neg_features, dtype=object)

    # convert negative feature values to plot values
    neg_val = out_value
    for i in neg_features:
        val = float(i[0])
        neg_val = neg_val + np.abs(val)
        i[0] = neg_val
    if len(neg_features) > 0:
        total_neg = np.max(neg_features[:, 0].astype(float)) - np.min(
            neg_features[:, 0].astype(float),
        )
    else:
        total_neg = 0

    # convert positive feature values to plot values
    pos_val = out_value
    for i in pos_features:
        val = float(i[0])
        pos_val = pos_val - np.abs(val)
        i[0] = pos_val

    if len(pos_features) > 0:
        total_pos = np.max(pos_features[:, 0].astype(float)) - np.min(
            pos_features[:, 0].astype(float),
        )
    else:
        total_pos = 0

    return pos_features, neg_features, total_pos, total_neg


def _add_bars(
    ax: plt.Axes,
    out_value: float,
    pos_features: np.ndarray,
    neg_features: np.ndarray,
) -> None:
    """Add bars to the plot.

    Args:
        ax: Axes of the plot
        out_value: grand total value
        pos_features: positive features
        neg_features: negative features
    """
    width_bar = 0.1
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200
    # Create bar for negative shap values
    rectangle_list, separator_list = _create_bars(
        out_value,
        neg_features,
        "negative",
        width_separators,
        width_bar,
    )
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    # Create bar for positive shap values
    rectangle_list, separator_list = _create_bars(
        out_value,
        pos_features,
        "positive",
        width_separators,
        width_bar,
    )
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)


def draw_higher_lower_element(
    out_value: float,
    offset_text: float,
) -> None:
    plt.text(
        out_value - offset_text,
        0.35,
        "higher",
        fontsize=13,
        color="#FF0D57",
        horizontalalignment="right",
    )
    plt.text(
        out_value + offset_text,
        0.35,
        "lower",
        fontsize=13,
        color="#1E88E5",
        horizontalalignment="left",
    )
    plt.text(
        out_value,
        0.34,
        r"$\leftarrow$",
        fontsize=13,
        color="#1E88E5",
        horizontalalignment="center",
    )
    plt.text(
        out_value,
        0.36,
        r"$\rightarrow$",
        fontsize=13,
        color="#FF0D57",
        horizontalalignment="center",
    )


def _draw_force_plot(
    interaction_value: InteractionValues,
    feature_names: np.ndarray,
    *,
    figsize: tuple[int, int],
    min_perc: float = 0.05,
    draw_higher_lower: bool = True,
) -> plt.Figure:
    """Draw the force plot.

    Note:
        The functionality was taken and adapted from the [SHAP package](https://github.com/shap/shap/blob/master/shap/plots/_force.py)
        which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).
        Do not use this function directly, use the ``force_plot`` function instead.

    Args:
        interaction_value: The interaction values to be plotted.
        feature_names: The names of the features.
        figsize: The size of the figure.
        min_perc: minimal percentage of the total effect that a feature must contribute to be shown.
            Defaults to ``0.05``.
        draw_higher_lower: Whether to draw the higher and lower indicator. Defaults to ``True``.

    Returns:
        The figure of the plot.

    """
    # turn off interactive plot
    plt.ioff()

    # compute overall metrics
    base_value = interaction_value.baseline_value
    out_value = np.sum(interaction_value.values)  # Sum of all values with the baseline value

    # split features into positive and negative values
    features_to_names = {i: str(name) for i, name in enumerate(feature_names)}
    pos_features, neg_features, total_pos, total_neg = _split_features(
        interaction_value.dict_values,
        features_to_names,
        out_value,
    )

    # define plots
    offset_text = (np.abs(total_neg) + np.abs(total_pos)) * 0.04

    fig, ax = plt.subplots(figsize=figsize)

    # compute axis limit
    update_axis_limits(ax, total_pos, pos_features, total_neg, neg_features, base_value, out_value)

    # add the bars to the plot
    _add_bars(ax, out_value, pos_features, neg_features)

    # add labels
    total_effect = np.abs(total_neg) + total_pos
    fig, ax = _add_labels(
        fig,
        ax,
        out_value,
        neg_features,
        "negative",
        offset_text,
        total_effect,
        min_perc=min_perc,
        text_rotation=0,
    )

    fig, ax = _add_labels(
        fig,
        ax,
        out_value,
        pos_features,
        "positive",
        offset_text,
        total_effect,
        min_perc=min_perc,
        text_rotation=0,
    )

    # add higher and lower element
    if draw_higher_lower:
        draw_higher_lower_element(out_value, offset_text)

    # add label for base value
    _add_base_value(base_value, ax)

    # add output label
    out_names = ""
    _add_output_element(out_names, out_value, ax)

    # fix the whitespace around the plot
    plt.tight_layout()

    return plt.gcf()


def force_plot(
    interaction_values: InteractionValues,
    *,
    feature_names: np.ndarray | None = None,
    abbreviate: bool = True,
    show: bool = False,
    figsize: tuple[int, int] = (15, 4),
    draw_higher_lower: bool = True,
    contribution_threshold: float = 0.05,
) -> plt.Figure | None:
    """Draws a force plot for the given interaction values.

    Args:
        interaction_values: The ``InteractionValues`` to be plotted.
        feature_names: The names of the features. If ``None``, the features are named by their index.
        show: Whether to show or return the plot. Defaults to ``False`` and returns the plot.
        abbreviate: Whether to abbreviate the feature names. Defaults to ``True.``
        figsize: The size of the figure. Defaults to ``(15, 4)``.
        draw_higher_lower: Whether to draw the higher and lower indicator. Defaults to ``True``.
        contribution_threshold: Define the minimum percentage of the total effect that a feature
            must contribute to be shown in the plot. Defaults to 0.05.

    Returns:
        plt.Figure: The figure of the plot

    References:
        .. [1] SHAP is available at https://github.com/shap/shap

    """
    if feature_names is None:
        feature_names = [str(i) for i in range(interaction_values.n_players)]
    if abbreviate:
        feature_names = abbreviate_feature_names(feature_names)
    feature_names = np.array(feature_names)
    plot = _draw_force_plot(
        interaction_values,
        feature_names,
        figsize=figsize,
        draw_higher_lower=draw_higher_lower,
        min_perc=contribution_threshold,
    )
    if not show:
        return plot
    plt.show()
    return None
