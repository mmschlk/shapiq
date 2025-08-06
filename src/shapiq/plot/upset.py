"""This module contains the upset plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ._config import BLUE, RED

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapiq.interaction_values import InteractionValues


def upset_plot(
    interaction_values: InteractionValues,
    *,
    n_interactions: int = 20,
    feature_names: Sequence[str] | None = None,
    color_matrix: bool = False,
    all_features: bool = True,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> plt.Figure | None:
    """Plots the upset plot.

    UpSet plots[1]_ can be used to visualize the interactions between features. The plot consists of
    two parts: the upper part shows the interaction values as bars, and the lower part shows the
    interactions as a matrix. Originally, the UpSet plot was introduced by Lex et al. (2014)[1]_.
    For a more detailed explanation about the plots, see the references or the original
    [documentation](https://upset.app/).

    An example of this plot is shown below.

    .. image:: /_static/images/upset_plot.png
        :width: 600
        :align: center

    Args:
        interaction_values: The interaction values as an ``InteractionValues`` object.
        feature_names: The names of the features. Defaults to ``None``. If ``None``, the features
            will be named with their index.
        n_interactions: The number of top interactions to plot. Defaults to ``20``. Note this number
            is completely arbitrary and can be adjusted to the user's needs.
        color_matrix: Whether to color the matrix (red for positive values, blue for negative) or
            not (black). Defaults to ``False``.
        all_features: Whether to plot all ``n_players`` features or only the features that are
            present in the top interactions. Defaults to ``True``.
        figsize: The size of the figure. Defaults to ``None``. If ``None``, the size will be set
            automatically depending on the number of features.
        show: Whether to show the plot. Defaults to ``False``.

    Returns:
        If ``show`` is ``True``, the function returns ``None``. Otherwise, it returns a tuple with
        the figure and the axis of the plot.

    References:
        .. [1] Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot, Hanspeter Pfister. UpSet: Visualization of Intersecting Sets IEEE Transactions on Visualization and Computer Graphics (InfoVis), 20(12): 1983--1992, doi:10.1109/TVCG.2014.2346248, 2014.

    """
    # prepare data ---------------------------------------------------------------------------------
    values = interaction_values.values
    values_ids: dict[int, tuple[int, ...]] = {
        v: k for k, v in interaction_values.interaction_lookup.items()
    }
    values_abs = abs(values)
    idx = values_abs.argsort()[::-1]
    idx = idx[:n_interactions] if n_interactions > 0 else idx
    values = values[idx]
    interactions: list[tuple[int, ...]] = [values_ids[i] for i in idx]

    # prepare feature names ------------------------------------------------------------------------
    if all_features:
        features = set(range(interaction_values.n_players))
    else:
        features = {feature for interaction in interactions for feature in interaction}
    n_features = len(features)
    feature_pos = {feature: n_features - 1 - i for i, feature in enumerate(features)}
    if feature_names is None:
        feature_names = [f"Feature {feature}" for feature in features]
    else:
        feature_names = [feature_names[feature] for feature in features]

    # create figure --------------------------------------------------------------------------------
    height_upper, height_lower = 5, n_features * 0.75
    height = height_upper + height_lower
    ratio = [height_upper, height_lower]
    if figsize is None:
        figsize = (10, height)
    else:
        if figsize[1] is None:
            figsize = (figsize[0], height)
        if figsize[0] is None:
            figsize = (10, figsize[1])

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": ratio}, sharex=True)

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
        y_pos = [feature_pos[feature] for feature in interaction]
        ax[1].plot(
            [x_pos for _ in range(len(interaction))],
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
    ax[0].axhline(0, color="black", linewidth=0.5)  # add line at 0

    # beautify lower plot --------------------------------------------------------------------------
    ax[1].set_ylim(-1, n_features)
    ax[1].yaxis.set_ticks(range(n_features))
    ax[1].set_yticklabels(reversed(feature_names))
    ax[1].tick_params(axis="y", length=0)  # remove y-ticks
    ax[1].set_xticks([])  # remove x-axis
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    # background shading
    for i in range(n_features):
        if i % 2 == 0:
            ax[1].axhspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.25, zorder=0, lw=0)

    # adjust whitespace
    plt.subplots_adjust(hspace=0.0)

    if not show:
        return fig
    plt.show()
    return None
