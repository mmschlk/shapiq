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

from shapiq.interaction_values import InteractionValues, aggregate_interaction_values

from .utils import abbreviate_feature_names


def _lch2rgb(lch_ab: np.ndarray | list) -> np.ndarray:
    """Converts LCh color space to RGB (sRGB) color space.

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
    return lab2rgb(lab)


def _get_red_blue_cmap() -> mcolors.LinearSegmentedColormap:
    """Creates a red-blue colormap with a smooth transition from blue to red.

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


def _get_config(row_height: float) -> dict:
    """Returns the configuration for the beeswarm plot.

    Args:
        row_height (float): Height of each row in the plot.

    Returns:
        dict: Configuration dictionary.
    """
    config_dict = {
        "dot_size": 10,
        "margin_y": 0.01,
        "color_nan": "#777777",
        "color_lines": "#cccccc",
        "color_rectangle": "#eeeeee",
        "alpha_rectangle": 0.5,
    }
    margin = max(-0.1875 * row_height + 0.3875, 0.15)  # 0.35 if row_height < 1 else 0.2
    margin_label = 0.5 - min(row_height / 3, 0.2)
    config_dict["margin_plot"] = margin
    config_dict["margin_label"] = margin_label
    config_dict["fontsize_ys"] = 10 if row_height <= 0.2 else 11
    return config_dict


def _beeswarm(interaction_values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Creates vertical offsets for a beeswarm plot.

    Args:
        interaction_values (np.ndarray): interaction values for a given feature.
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        np.ndarray: Vertical offsets (ys) for each point.
    """
    num_interactions = len(interaction_values)
    nbins = 100
    quant = np.round(
        nbins
        * (interaction_values - np.min(interaction_values))
        / (np.max(interaction_values) - np.min(interaction_values) + 1e-9)
    )

    inds = np.argsort(quant + rng.uniform(-1e-6, 1e-6, num_interactions))

    layer = 0
    last_bin = -1
    ys = np.zeros(num_interactions)
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    return ys


def _calculate_range(num_sub_features: int, i: int, margin: float) -> tuple[float, float]:
    """Calculates the y-axis range for a given sub-feature index in a beeswarm plot.

    Args:
        num_sub_features (int): Total number of sub-features in the interaction.
        i (int): Index of the current sub-feature.
        margin (float): Margin to apply to the y-axis range.

    Returns:
        tuple: A tuple containing the minimum and maximum y-axis values for the sub-feature.
    """
    if num_sub_features > 1:
        if i == 0:
            y_min = margin / 2 - 0.5
            y_max = 0.5 - margin / 4
        elif i == num_sub_features - 1:
            y_min = margin / 4 - 0.5
            y_max = 0.5 - margin / 2
        else:
            y_min = margin / 4 - 0.5
            y_max = 0.5 - margin / 4
    else:
        y_min = margin / 2 - 0.5
        y_max = 0.5 - margin / 2
    return y_min, y_max


def beeswarm_plot(
    interaction_values_list: list[InteractionValues],
    data: pd.DataFrame | np.ndarray,
    *,
    max_display: int | None = 10,
    feature_names: list[str] | None = None,
    abbreviate: bool = True,
    alpha: float = 0.8,
    row_height: float = 0.4,
    ax: plt.Axes | None = None,
    rng_seed: int | None = 42,
    show: bool = True,
) -> plt.Axes | None:
    """Plots a beeswarm plot of SHAP-IQ interaction values.

    Args:
        interaction_values_list (list[InteractionValues]): List of SHAP-IQ interaction values.
        data (pd.DataFrame | np.ndarray): The input data used to compute the interaction values.
        max_display (int): Maximum number of interactions to display. Defaults to 10.
        feature_names (list[str] | None): Names of the features. If None, default names will be used.
        abbreviate (bool): Whether to abbreviate feature names. Defaults to True.
        alpha (float): Transparency level of the points. Defaults to 0.8.
        row_height (float): Height of each row in the plot. Defaults to 0.4.
        ax (plt.Axes | None): Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        rng_seed (int | None): Random seed for reproducibility. Defaults to 42.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        plt.Axes | None: The Axes object if `show` is False, otherwise None.
    """
    if not isinstance(interaction_values_list, list) or len(interaction_values_list) == 0:
        error_message = "shap_interaction_values must be a non-empty list."
        raise ValueError(error_message)
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        error_message = f"X must be a pandas DataFrame or a numpy array. Got: {type(data)}."
        raise TypeError(error_message)
    if len(interaction_values_list) != len(data):
        error_message = "Length of shap_interaction_values must match number of rows in X."
        raise ValueError(error_message)
    if row_height < 0:
        error_message = "row_height must be a non-negative value."
        raise ValueError(error_message)
    if alpha < 0:
        error_message = "alpha must be a non-negative value."
        raise ValueError(error_message)

    n_samples = len(data)
    n_players = interaction_values_list[0].n_players

    if feature_names is not None:
        if abbreviate:
            feature_names = abbreviate_feature_names(feature_names)
    else:
        feature_names = ["F" + str(i) for i in range(n_players)]

    if len(feature_names) != n_players:
        error_message = "Length of feature_names must match n_players."
        raise ValueError(error_message)

    feature_mapping = dict(enumerate(feature_names))

    list_of_abs_interaction_values = [abs(iv) for iv in interaction_values_list]
    global_values = aggregate_interaction_values(
        list_of_abs_interaction_values, aggregation="mean"
    )  # to match the order in bar plots

    interaction_keys = list(global_values.interaction_lookup.keys())
    all_global_interaction_vals = global_values.values  # noqa: PD011 # since ruff thinks this is a dataframe
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

    x_numpy = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else data.astype(float)

    shap_values_dict = {}

    for interaction in interactions_to_plot:
        shap_values_dict[interaction] = np.array(
            [sv.dict_values[interaction] for sv in interaction_values_list]
        )

    total_sub_features = sum(len(inter) for inter in interactions_to_plot)
    if ax is None:
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
        row_height = (fig.get_size_inches()[1] - 1.5) / total_sub_features
    config_dict = _get_config(row_height)

    cmap = _get_red_blue_cmap()

    y_level = 0  # start plotting from the bottom
    y_tick_labels_formatted = {"y": [], "text": []}
    h_lines = []  # horizontal lines between interaction groups
    rectangles = []

    margin_label = config_dict["margin_label"]
    # iterate through interactions in reverse order for plotting (bottom-up)
    for interaction_index, interaction in enumerate(reversed(interactions_to_plot)):
        num_sub_features = len(interaction)

        if interaction_index % 2 == 0:
            bottom_y = y_level - 0.5
            height = num_sub_features
            if bottom_y == -0.5:
                bottom_y -= config_dict["margin_y"]
                height += config_dict["margin_y"]
            rectangles.append((bottom_y, height))

        group_midpoint_y = y_level + (num_sub_features - 1) / 2.0
        num_labels = num_sub_features + max(num_sub_features - 1, 0)
        bottom_y = group_midpoint_y - margin_label * (num_labels - 1) / 2
        upper_y = group_midpoint_y + margin_label * (num_labels - 1) / 2
        positions = (
            np.linspace(bottom_y, upper_y, num_labels)
            if num_sub_features > 1
            else np.array([group_midpoint_y])
        )
        j = 0
        for i, label in enumerate(reversed(interaction)):
            lb = feature_mapping[label]
            current_group_midpoint = positions[i + j]

            y_tick_labels_formatted["y"].append(current_group_midpoint)
            y_tick_labels_formatted["text"].append(lb)

            if i < num_sub_features - 1:
                y_tick_labels_formatted["y"].append(positions[i + j + 1])
                y_tick_labels_formatted["text"].append("x")
                j += 1

        # add horizontal lines
        if 0 < interaction_index < len(interactions_to_plot) - 1:
            upper_point = group_midpoint_y - num_sub_features / 2.0
            lower_point = group_midpoint_y + num_sub_features / 2.0
            h_lines.append(upper_point)
            h_lines.append(lower_point)

        current_shap_values = shap_values_dict[interaction]

        # calculate beeswarm offsets
        ys_raw = _beeswarm(current_shap_values, rng=np.random.default_rng(rng_seed))
        for i, sub_feature_idx in enumerate(interaction):
            y_min, y_max = _calculate_range(num_sub_features, i, config_dict["margin_plot"])
            range_y = np.max(ys_raw) - np.min(ys_raw) if np.max(ys_raw) != np.min(ys_raw) else 1.0
            ys = y_min + (ys_raw - np.min(ys_raw)) * (y_max - y_min) / range_y
            feature_values = x_numpy[:, sub_feature_idx]

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
                x=current_shap_values[nan_mask],
                y=y_level + ys[nan_mask],
                color=config_dict["color_nan"],
                s=config_dict["dot_size"],
                alpha=alpha * 0.5,
                linewidth=0,
                rasterized=n_samples > 500,
                zorder=2,
            )

            # valid points
            ax.scatter(
                x=current_shap_values[valid_mask],
                y=y_level + ys[valid_mask],
                c=feature_values[valid_mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=config_dict["dot_size"],
                alpha=alpha,
                linewidth=0,
                rasterized=n_samples > 500,
                zorder=2,
            )
            y_level += 1

    # add horizontal grid lines between interaction groups
    h_lines = list(set(h_lines))
    for y_line in h_lines:
        ax.axhline(
            y=y_line,
            color=config_dict["color_lines"],
            linestyle="--",
            linewidth=0.5,
            alpha=0.8,
            zorder=-1,
        )

    ax.xaxis.grid(
        visible=True,
        color=config_dict["color_lines"],
        linestyle="--",
        linewidth=0.5,
        alpha=0.8,
        zorder=-1,
    )

    ax.axvline(x=0, color="#999999", linestyle="-", linewidth=1, zorder=1)
    ax.set_axisbelow(True)

    ax.set_xlabel("SHAP-IQ Interaction Value (impact on model output)", fontsize=12)
    ax.set_ylabel("")

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)

    xlims = ax.get_xlim()
    for y_coords in rectangles:
        bottom_y, height = y_coords

        x_left, x_right = xlims[0], xlims[1]
        rect = plt.Rectangle(
            (x_left, bottom_y),
            x_right - x_left,
            height,
            facecolor=config_dict["color_rectangle"],
            edgecolor=config_dict["color_rectangle"],
            alpha=config_dict["alpha_rectangle"],
            zorder=-3,
        )
        ax.add_patch(rect)

    ax.set_yticks(y_tick_labels_formatted["y"])
    ax.set_yticklabels(y_tick_labels_formatted["text"], fontsize=config_dict["fontsize_ys"])

    ax.set_ylim(-0.5 - config_dict["margin_y"], y_level - 0.5 + config_dict["margin_y"])

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
    return None
