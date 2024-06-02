"""This module contains the plotting utilities for the benchmark results."""

from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: add the plot colors and styles for different approximators as well
STYLE_DICT: dict[str, dict[str, str]] = {
    # permutation sampling
    "PermutationSamplingSII": {"color": "#7d53de", "marker": "o"},
    "PermutationSamplingSTII": {"color": "#7d53de", "marker": "o"},
    "PermutationSamplingSV": {"color": "#7d53de", "marker": "o"},
    # KernelSHAP-IQ
    "KernelSHAP": {"color": "#ff6f00", "marker": "o"},
    "KernelSHAPIQ": {"color": "#ff6f00", "marker": "o"},
    # inconsistent KernelSHAP-IQ
    "InconsistentKernelSHAPIQ": {"color": "#ffba08", "marker": "o"},
    "kADDSHAP": {"color": "#ffba08", "marker": "o"},
    # SVARM-based
    "SVARMIQ": {"color": "#00b4d8", "marker": "o"},
    "SVARM": {"color": "#00b4d8", "marker": "o"},
    # shapiq
    "SHAPIQ": {"color": "#ef27a6", "marker": "o"},
    "UnbiasedKernelSHAP": {"color": "#ef27a6", "marker": "o"},
    # misc SV
    "OwenSamplingSV": {"color": "#7DCE82", "marker": "o"},
    "StratifiedSamplingSV": {"color": "#4B7B4E", "marker": "o"},
}
STYLE_DICT = defaultdict(lambda: {"color": "black", "marker": "o"}, STYLE_DICT)
MARKERS = []
LIGHT_GRAY = "#d3d3d3"
LINE_STYLES_ORDER = {0: "solid", 1: "dotted", 2: "solid", 3: "dashed", 4: "dashdot", "all": "solid"}
LINE_MARKERS_ORDER = {0: "o", 1: "o", 2: "s", 3: "X", 4: "d", "all": "o"}
LINE_THICKNESS = 1.5
MARKER_SIZE = 7


LOG_SCALE_MIN = 1e-7


def get_game_title_name(game_name: str) -> str:
    """Changes the game name to a more readable title.

    Args:
        game_name: The game name to change.

    Returns:
        The game title name.

    Example:
        >>> get_game_title_name("ImageClassifierLocalXAI")
        "Image Classifier Local XAI"
        >>> get_game_title_name("AdultCensusClusterExplanation")
        "Adult Census Cluster Explanation"
    """
    # split words by capital letters
    words = ""
    for char in game_name:
        if char.isupper():
            words += " "
        words += char
    words = words.replace("Tree S H A P I Q", "TreeSHAPIQ")  # TreeSHAPIQ
    words = words.replace("X A I", "XAI")  # XAI
    words = words.replace("S O U M", "SOUM")  # SOUM
    return words.strip()


def agg_percentile(q: float) -> Callable[[np.ndarray], float]:
    """Get the aggregation function for the given percentile.

    Args:
        q: The percentile to compute.

    Returns:
        The aggregation function.
    """

    def quantile(x) -> float:
        """Performs the aggregation function for the given percentile."""
        return np.percentile(x, q)

    quantile.__name__ = f"quantile_{q}"
    return quantile


def plot_approximation_quality(
    data: pd.DataFrame,
    metric: str = "MSE",
    orders: Optional[list[Union[int, str]]] = None,
    approximators: Optional[list[str]] = None,
    aggregation: str = "mean",
    confidence_metric: Optional[str] = "sem",
    log_scale_y: bool = False,
    log_scale_min: float = LOG_SCALE_MIN,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the approximation quality curves.

    Args:
        data: The data to plot the values from.
        metric: The metric to plot. Defaults to "MSE".
        orders: The orders to plot. If `None`, all orders are plotted. Defaults to `None`.
            Can be a list of integers or a single integer.
        approximators: The approximators to plot. Defaults to `None`. When `None`, all approximators
            are plotted.
        aggregation: The aggregation function to plot for the metric values. Defaults to "mean".
            Available options are "mean", "median", "quantile_95", "quantile_5".
        confidence_metric: The metric to use for the confidence interval. Defaults to "sem".
            Available options are "sem", "std", "var", "quantile_95", "quantile_5".
        log_scale_y: Whether to use a log scale for the y-axis. Defaults to `False`.

    Returns:
        The figure and axes of the plot.
    """
    # get the metric data
    metric_data = get_metric_data(data, metric)

    # make sure orders is a list
    if orders is None:
        orders = ["all"]
    if isinstance(orders, int) or isinstance(orders, str):
        orders = [orders]

    # make sure approximators is a list
    if approximators is None:
        approximators = list(metric_data["approximator"].unique())
        print("Approximators:", approximators)

    # set the confidence metrics
    confidence_metric_low, confidence_metric_high = confidence_metric, confidence_metric
    if confidence_metric is not None and "quantile" in confidence_metric:
        confidence_metric_low = "quantile_5"
        confidence_metric_high = "quantile_95"

    # create the plot
    fig, ax = plt.subplots()
    for approximator in approximators:
        for order in metric_data["order"].unique():
            if orders is not None and order not in orders:
                continue
            data_order = metric_data[
                (metric_data["approximator"] == approximator) & (metric_data["order"] == order)
            ].copy()

            if log_scale_y:
                # manually set all below log_scale_min to log_scale_min without a lambda function
                data_order[aggregation] = data_order[aggregation].apply(
                    lambda x: log_scale_min if x < log_scale_min else x
                )

            # get the plot colors and styles
            line_style, line_marker = LINE_STYLES_ORDER[order], LINE_MARKERS_ORDER[order]
            color = STYLE_DICT[approximator]["color"]

            # plot the mean values
            ax.plot(
                data_order["used_budget"],
                data_order[aggregation],
                color=color,
                linestyle=line_style,
                marker=line_marker,
                linewidth=LINE_THICKNESS,
                mec="white",
                markersize=MARKER_SIZE,
            )
            # plot the error bars if the confidence metric is not None
            if confidence_metric is not None:
                ax.fill_between(
                    data_order["used_budget"],
                    data_order[aggregation] - data_order[confidence_metric_low],
                    data_order[aggregation] + data_order[confidence_metric_high],
                    alpha=0.1,
                    color=color,
                )

    # add %model calls to the x-axis as a secondary axis
    _set_x_axis_ticks(ax, n_players=int(data["n_players"].unique().max()))

    # add x/y labels
    ax.set_ylabel(metric)
    ax.set_xlabel(r"Model Evaluations (relative to $2^n$)")

    # add grid to x-axis
    ax.grid(axis="x", color=LIGHT_GRAY, linestyle="dashed")

    if log_scale_y:
        _set_y_axis_log_scale(ax, log_scale_min)

    return fig, ax


def _set_x_axis_ticks(ax: plt.Axes, n_players: int) -> None:
    """Sets the x-axis ticks in 25% intervals."""
    if n_players <= 16:  # only for small number of players set the ticks as 25% intervals
        budgets_relative = np.arange(0, 1.25, 0.25)
        budgets = budgets_relative * (2**n_players)
    else:
        budgets = ax.get_xticks()
        budgets_relative = budgets / (2**n_players)

    xtick_labels = []
    for bdgt, bdgt_rel in zip(budgets, budgets_relative):
        bdgt_rel_str = f"{bdgt_rel:.0%}"
        if bdgt_rel <= 0.01 and bdgt_rel != 0:
            bdgt_rel_str = "<1%"
        if bdgt_rel == 0:
            xtick_labels.append("0")
        else:
            xtick_labels.append(f"{int(bdgt)}\n({bdgt_rel_str})")

    ax.set_xticks(budgets)
    ax.set_xticklabels(xtick_labels)


def _set_y_axis_log_scale(ax: plt.Axes, log_scale_min: float) -> None:
    """Sets the y-axis to a log scale and adjusts the limits."""
    # adjust the top limit to be one order of magnitude higher than the current top limit
    top_lim = ax.get_ylim()[1]
    top_lim = f"{top_lim:.2e}"  # get the top limi in scientific notation
    top_lim = top_lim.split("e")[1]  # get the exponent
    top_lim = int(top_lim) + 1  # get the top limit as the exponent + 1
    top_lim = 10**top_lim  # get the top limit in scientific notation

    # set the y-axis limits
    ax.set_ylim(top=top_lim)
    ax.set_ylim(bottom=log_scale_min)
    ax.set_yscale("log")


def get_metric_data(results_df: pd.DataFrame, metric: str = "MSE") -> pd.DataFrame:
    """Get the metric data for the given results.

    Args:
        results_df: The results dataframe.
        metric: The metric to get the data for. Defaults to "MSE".

    Returns:
        The metric data.
    """

    # get the metric columns for each order in the results
    metric_columns = [col for col in results_df.columns if metric in col]

    metric_dfs = []
    for metric_col in metric_columns:
        data_order = (
            results_df.groupby(["approximator", "used_budget", "iteration"])
            .agg(
                {
                    metric_col: [
                        "mean",
                        "std",
                        "var",
                        "count",
                        "median",
                        agg_percentile(95),
                        agg_percentile(5),
                    ]
                }
            )
            .reset_index()
        )
        data_order["order"] = "all" if "_" not in metric_col else int(metric_col.split("_")[0])
        # rename the columns of grouped data
        new_columns = [
            "_".join(col).strip() if col[1] != "" else col[0] for col in data_order.columns
        ]
        new_columns = [col.replace(f"{metric_col}_", "") for col in new_columns]

        data_order.columns = new_columns
        metric_dfs.append(data_order)

    # concat the dataframes along the row
    metric_df = pd.concat(metric_dfs)

    # compute the standard error
    metric_df["sem"] = metric_df["std"] / metric_df["count"] ** 0.5  # compute standard error

    return metric_df


def add_legend(
    axis: plt.Axes,
    approximators: list[str],
    orders: Optional[list[Union[int, str]]] = None,
    legend_subtitle: bool = True,
    loc: str = "best",
) -> None:
    """Add the legend to the plot.

    Args:
        axis: The axes of the plot.
        approximators: The list of approximators to add to the legend.
        orders: The orders to add to the legend. Can be a list of integers or a single integer.
            Defaults to `None`.
        legend_subtitle: Whether to add a subtitle to the legend. Defaults to `True`.
        loc: The location of the legend. Defaults to "upper right".
    """
    if orders is None and approximators is None:
        return

    # plot the order elements
    if orders is not None:
        if isinstance(orders, int) or isinstance(orders, str):
            orders = [orders]
        if legend_subtitle:
            axis.plot([], [], label="$\\bf{Order}$", color="none")
        for order in orders:
            axis.plot(
                [],
                [],
                label=f"Order {order}",
                color="black",
                linestyle=LINE_STYLES_ORDER[order],
                marker=LINE_MARKERS_ORDER[order],
                linewidth=LINE_THICKNESS,
                mec="white",
            )

    # plot the approximator elements
    if legend_subtitle:
        axis.plot([], [], label="$\\bf{Method}$", color="none")
    for approximator in approximators:
        axis.plot(
            [],
            [],
            label=approximator,
            color=STYLE_DICT[approximator]["color"],
            linewidth=LINE_THICKNESS,
        )

    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc=loc)
