"""This module contains the upset plot."""

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt

from ..interaction_values import InteractionValues
from ._config import BLUE, RED


def upset_plot(
    interaction_values: InteractionValues,
    feature_names: Optional[Sequence[str]] = None,
    color_matrix: bool = False,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Plots the upset plot.

    UpSet plots are used to visualize the interactions between features. The plot consists of two
    parts: the upper part shows the interaction values as bars, and the lower part shows the
    interactions as a matrix. Originally, the UpSet plot was introduced by Lex et al. [1].

    Args:
        interaction_values: The interaction values as an interaction object.
        feature_names: The names of the features. Defaults to ``None``. If ``None``, the features
            will be named with their index.
        color_matrix: Whether to color the matrix (red for positive values, blue for negative) or
            not (black). Defaults to ``False``.
        show: Whether to show the plot. Defaults to ``False``.

    Returns:
        If ``show`` is ``True``, the function returns ``None``. Otherwise, it returns a tuple with
        the figure and the axis of the plot.

    References:
        - [1] Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot, Hanspeter Pfister. UpSet: Visualization of Intersecting Sets IEEE Transactions on Visualization and Computer Graphics (InfoVis), 20(12): 1983--1992, doi:10.1109/TVCG.2014.2346248, 2014.
    """

    # prepare data
    values = interaction_values.values
    values_ids: dict[int, tuple[int, ...]] = {
        v: k for k, v in interaction_values.interaction_lookup.items()
    }
    values_abs = abs(values)
    idx = values_abs.argsort()[::-1]
    values = values[idx]
    interactions: list[tuple[int, ...]] = [values_ids[i] for i in idx]
    features = set([feature for interaction in interactions for feature in interaction])
    n_features = len(features)
    feature_pos: dict[int, int] = {}
    for pos in range(n_features):
        feature_pos[list(features)[pos]] = n_features - pos - 1

    # create figure
    height_upper, height_lower = 5, n_features * 0.75
    height = height_upper + height_lower
    ratio = [height_upper, height_lower]
    fig, ax = plt.subplots(
        2, 1, figsize=(10, height), gridspec_kw={"height_ratios": ratio}, sharex=True
    )

    # plot lower part of the upset plot
    for x_pos, interaction in enumerate(interactions):
        color = RED.hex if values[x_pos] >= 0 else BLUE.hex

        # plot upper part
        bar = ax[0].bar(x_pos, values[x_pos], color=color)
        label = [f"{values[x_pos]:.2f}"]
        ax[0].bar_label(bar, label, label_type="edge", color="black", fontsize=12, padding=3)

        # plot lower part
        # plot the matrix in the background
        ax[1].plot(
            [x_pos for _ in range(n_features)],
            list(range(n_features)),
            color="lightgray",
            marker="o",
            markersize=15,
            linewidth=0,
        )
        # add the interaction to the matrix
        x_pos = [x_pos for _ in range(len(interaction))]
        y_pos = [feature_pos[feature] for feature in interaction]
        ax[1].plot(
            x_pos,
            y_pos,
            color="black" if not color_matrix else color,
            marker="o",
            markersize=15,
            linewidth=1.5,
        )

    # beautify upper plot --------------------------------------------------------------------------
    min_max = (min(values), max(values))
    delta = (min_max[1] - min_max[0]) * 0.1
    ax[0].set_ylim(min_max[0] - delta, min_max[1] + delta)
    ax[0].set_ylabel("Interaction Value")
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["bottom"].set_visible(False)
    # add line at 0
    ax[0].axhline(0, color="black", linewidth=0.5)

    # beautify lower plot
    ax[1].set_ylim(-1, n_features)
    # add feature names
    if feature_names is None:
        feature_names = [f"Feature {feature}" for feature in features]
    else:
        feature_names = [feature_names[feature] for feature in features]
    ax[1].yaxis.set_ticks(range(n_features))
    ax[1].set_yticklabels(feature_names)
    # remove y-tick markings but keep labels
    ax[1].tick_params(axis="y", length=0)
    # remove x-axis
    ax[1].set_xticks([])
    # remove spines
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)

    # add an alternating lightgray bacground behind the feature names
    for i in range(n_features):
        if i % 2 == 0:
            ax[1].axhspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.25, zorder=0, lw=0)

    # adjust whitespace
    plt.subplots_adjust(hspace=0.0)

    if not show:
        return fig
    plt.show()
