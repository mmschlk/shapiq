"""Wrapper for the beeswarm plot from the ``shap`` package.

Note:
    Code and implementation was taken and adapted from the [SHAP package](https://github.com/shap/shap)
    which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).

"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import lab2rgb, lch2lab

from ..interaction_values import InteractionValues, aggregate_interaction_values
from .utils import abbreviate_feature_names


def _lch2rgb(lch_ab: np.ndarray | list) -> np.ndarray:
    """
    Converts LCh color space to RGB (sRGB) color space.
    Args:
        lch_ab (np.array | list): LCh color space values, where the last dimension contains
            [L, C, h] values. The hue (h) should be in radians.
    Returns:
        np.ndarray: RGB color space values in sRGB format.
    """
    lch_ab = np.asanyarray(lch_ab)
    if lch_ab.shape[-1] != 3:
        error_message = (
            f"Input array must have 3 channels in the last dimension. Got shape: {lch_ab.shape}"
        )
        raise ValueError(error_message)
    lab = lch2lab(lch_ab)
    rgb_srgb = lab2rgb(lab)
    return rgb_srgb


def _get_red_blue_cmap() -> mcolors.LinearSegmentedColormap:
    """
    Creates a red-blue colormap with a smooth transition from blue to red.
    Returns:
        mcolors.LinearSegmentedColormap: A colormap object that transitions from blue to red.
    """
    blue_lch = [54.0, 70.0, 4.6588]
    l_mid = 40.0
    red_lch = [54.0, 90.0, 0.35470565 + 2 * np.pi]
    gray_lch = [55.0, 0.0, 0.0]

    gray_rgb = _lch2rgb(gray_lch)
    reds, greens, blues, alphas = [], [], [], []
    nsteps = 100
    l_vals = np.concatenate(
        [np.linspace(blue_lch[0], l_mid, nsteps // 2), np.linspace(l_mid, red_lch[0], nsteps // 2)]
    )
    c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
    h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)

    for i in range(nsteps):
        pos = i / (nsteps - 1) if nsteps > 1 else 0.0
        lch = np.array([l_vals[i], c_vals[i], h_vals[i]])
        rgb = _lch2rgb(lch)
        reds.append((pos, rgb[0], rgb[0]))
        greens.append((pos, rgb[1], rgb[1]))
        blues.append((pos, rgb[2], rgb[2]))
        alphas.append((pos, 1.0, 1.0))

    cdict = {"red": reds, "green": greens, "blue": blues, "alpha": alphas}
    red_blue = mcolors.LinearSegmentedColormap("red_blue_v3", cdict)
    red_blue.set_bad(gray_rgb.tolist(), 1.0)
    red_blue.set_over(gray_rgb.tolist(), 1.0)
    red_blue.set_under(gray_rgb.tolist(), 1.0)
    return red_blue


def _beeswarm(shaps: np.ndarray, row_height: float, rng: np.random.Generator | None = None):
    """
    Creates vertical offsets for a beeswarm plot.

    Args:
        shaps (np.ndarray): interaction values for a given feature.
        row_height (float): target height for the swarm distribution.
        rng (np.random.Generator, optional): Random number generator. Defaults to None - if None, a default RNG with seed 0 is used.
    Returns:
        np.ndarray: Vertical offsets (ys) for each point.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(shaps)
    nbins = 100
    quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-9))

    inds = np.argsort(quant + rng.uniform(-1e-6, 1e-6, N))

    layer = 0
    last_bin = -1
    ys = np.zeros(N)
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]

    max_abs_ys = np.max(np.abs(ys))
    if max_abs_ys > 0:
        multiplier = 0.7 if row_height < 1 else 0.9
        ys *= multiplier * (0.5 / (max_abs_ys + 1))
    return ys


def beeswarm_plot(
    shap_interaction_values: list[InteractionValues],
    data: pd.DataFrame | np.ndarray,
    max_display: int | None = 10,
    feature_names: list[str] | None = None,
    abbreviate: bool = True,
    jitter: bool = False,
    alpha: float = 0.8,
    show: bool = True,
    row_height: float = 0.4,
    ax: plt.Axes | None = None,
    rng_seed: int | None = 42,
) -> plt.Axes | None:
    """
    Plots a beeswarm plot of SHAP-IQ interaction values.
    Args:
        shap_interaction_values (list[InteractionValues]): List of SHAP-IQ interaction values.
        data (pd.DataFrame | np.ndarray): The input data used to compute the interaction values.
        max_display (int): Maximum number of interactions to display. Defaults to 5.
        feature_names (list[str] | None): Names of the features. If None, default names will be used.
        abbreviate (bool): Whether to abbreviate feature names. Defaults to True.
        jitter (bool): Whether to add jitter to the points. Defaults to False.
        alpha (float): Transparency level of the points. Defaults to 0.8.
        show (bool): Whether to show the plot. Defaults to True.
        row_height (float): Height of each row in the plot. Defaults to 0.4.
        ax (plt.Axes | None): Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        rng_seed (int | None): Random seed for reproducibility. Defaults to 42.
    Returns:
        plt.Axes | None: The Axes object if `show` is False, otherwise None.
    """
    if not isinstance(shap_interaction_values, list) or len(shap_interaction_values) == 0:
        error_message = "shap_interaction_values must be a non-empty list."
        raise ValueError(error_message)
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        error_message = f"X must be a pandas DataFrame or a numpy array. Got: {type(data)}."
        raise ValueError(error_message)
    if len(shap_interaction_values) != len(data):
        error_message = "Length of shap_interaction_values must match number of rows in X."
        raise ValueError(error_message)

    dot_size = 10

    n_samples = len(data)
    n_players = shap_interaction_values[0].n_players

    if feature_names is not None:
        if abbreviate:
            feature_names = abbreviate_feature_names(feature_names)
    else:
        feature_names = ["F" + str(i) for i in range(n_players)]

    if len(feature_names) != n_players:
        error_message = "Length of feature_names must match n_players."
        raise ValueError(error_message)

    feature_mapping = dict(enumerate(feature_names))
    rng = np.random.default_rng(rng_seed)

    list_of_abs_interaction_values = [abs(iv) for iv in shap_interaction_values]
    global_values = aggregate_interaction_values(
        list_of_abs_interaction_values, aggregation="mean"
    )  # to match the order in bar plots

    interaction_keys = list(global_values.interaction_lookup.keys())
    all_global_interaction_vals = global_values.values
    if interaction_keys[0] == ():  # check for base value
        interaction_keys = interaction_keys[1:]
        all_global_interaction_vals = all_global_interaction_vals[1:]

    # Sort interactions by aggregated importance
    feature_order = np.argsort(all_global_interaction_vals)[::-1]
    if max_display is None:
        max_display = len(feature_order)
    num_interactions_to_display = min(max_display, len(feature_order))
    feature_order = feature_order[:num_interactions_to_display]

    interactions_to_plot = [interaction_keys[i] for i in feature_order]

    if isinstance(data, pd.DataFrame):
        X_numpy = data.to_numpy(dtype=float)
    else:
        X_numpy = data.astype(float)

    shap_values_dict = {}
    for interaction in interactions_to_plot:
        shap_values_dict[interaction] = np.array(
            [sv.dict_values[interaction] for sv in shap_interaction_values]
        )

    if ax is None:
        total_sub_features = sum(len(inter) for inter in interactions_to_plot)
        fig_height = total_sub_features * row_height + 1.5
        fig_width = 8 + 0.3 * max(
            [
                np.max([len(feature_mapping[f]) for f in interaction])
                for interaction in interactions_to_plot
            ]
        )
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(fig_width, fig_height)
    else:
        fig = ax.get_figure()

    cmap = _get_red_blue_cmap()

    y_level = 0  # start plotting from the bottom
    y_tick_positions = []
    y_tick_labels_formatted = []
    h_lines = []  # horizontal lines between interaction groups

    min_x = np.min([shap_values_dict[interaction] for interaction in interactions_to_plot])
    max_x = np.max([shap_values_dict[interaction] for interaction in interactions_to_plot])
    range_x = np.abs(max_x - min_x)

    # iterate through interactions in reverse order for plotting (bottom-up)
    for interaction_index, interaction in enumerate(reversed(interactions_to_plot)):
        num_sub_features = len(interaction)

        group_midpoint_y = y_level + (num_sub_features - 1) / 2.0
        for i, label in enumerate(reversed(interaction)):
            lb = feature_mapping[label]
            current_group_midpoint = y_level + i
            y_tick_positions.append(current_group_midpoint)
            y_tick_labels_formatted.append(lb)

            if i < num_sub_features - 1:
                y_tick_positions.append(current_group_midpoint + 0.5)
                y_tick_labels_formatted.append(" x ")

        # add horizontal lines
        if 0 < interaction_index < len(interactions_to_plot) - 1:
            upper_point = group_midpoint_y - num_sub_features / 2.0
            lower_point = group_midpoint_y + num_sub_features / 2.0
            h_lines.append(upper_point)
            h_lines.append(lower_point)

        current_shap_values = shap_values_dict[interaction]

        # calculate beeswarm offsets
        ys = _beeswarm(current_shap_values, row_height, rng=rng)

        if jitter:
            xs = current_shap_values + rng.uniform(-range_x * 0.005, range_x * 0.005, n_samples)
        else:
            xs = current_shap_values

        for sub_feature_idx in interaction:
            feature_values = X_numpy[:, sub_feature_idx]

            # nan handling - plotting as gray
            nan_mask = np.isnan(feature_values)
            valid_mask = ~nan_mask

            valid_feature_values = feature_values[valid_mask]
            if len(valid_feature_values) > 0:
                vmin = np.min(valid_feature_values)
                vmax = np.max(valid_feature_values)
            else:
                vmin = 0
                vmax = 1
            if vmin == vmax:
                vmin -= 1e-9
                vmax += 1e-9

            ax.scatter(
                x=xs[nan_mask],
                y=y_level + ys[nan_mask],
                color="#777777",
                s=dot_size,
                alpha=alpha * 0.5,
                linewidth=0,
                rasterized=n_samples > 500,
            )

            # valid points
            ax.scatter(
                x=xs[valid_mask],
                y=y_level + ys[valid_mask],
                c=feature_values[valid_mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=dot_size,
                alpha=alpha,
                linewidth=0,
                rasterized=n_samples > 500,
            )
            y_level += 1

    ax.axvline(x=0, color="#999999", linestyle="-", linewidth=1, zorder=-1)

    # add horizontal grid lines between interaction groups
    h_lines = list(set(h_lines))
    for y_line in h_lines:
        ax.axhline(y=y_line, color="#cccccc", linestyle="--", linewidth=0.5, alpha=0.8, zorder=-1)

    ax.xaxis.grid(True, color="#cccccc", linestyle="--", linewidth=0.5, alpha=0.8, zorder=-1)
    ax.set_axisbelow(True)

    ax.set_xlabel("SHAP-IQ Interaction Value (impact on model output)", fontsize=12)
    ax.set_ylabel("")

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels_formatted, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)

    ax.set_ylim(-0.55, y_level - 0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    cb = fig.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value", size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)

    plt.tight_layout(rect=(0, 0, 0.95, 1))

    if not show:
        return ax
    plt.show()
