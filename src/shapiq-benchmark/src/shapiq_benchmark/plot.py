"""This module contains the plotting utilities for the benchmark results."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import scienceplots

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.approximator.base import Approximator

__all__ = [
    "abbreviate_application_name",
    "create_application_name",
    "get_game_title_name",
    "plot_approximation_quality",
]


STYLE_DICT: dict[str, dict[str, str]] = {
    # permutation sampling
    "PermutationSamplingSII": {
        "color": "#252525",
        "marker": None,
    },
    "PermutationSamplingSTII": {
        "color": "#252525",
        "marker": None,
    },
    "PermutationSamplingSV": {
        "color": "#252525",
        "marker": None,
    },
    # KernelSHAP-IQ
    "KernelSHAP": {"color": "#ff6f00", "marker": None, "linestyle": (0, (5, 1))},
    "KernelSHAPIQ": {"color": "#ff6f00", "marker": "o"},
    # inconsistent KernelSHAP-IQ
    "InconsistentKernelSHAPIQ": {"color": "#ffba08", "marker": "o"},
    "kADDSHAP": {"color": "#ffba08", "marker": "o"},
    # SVARM-based
    "SVARMIQ": {"color": "#707070", "marker": None},
    "SVARM": {"color": "#00b4d8", "marker": "o"},
    # shapiq
    "SHAPIQ": {"color": "#959595", "marker": None},
    "UnbiasedKernelSHAP": {"color": "#ef27a6", "marker": "o"},
    # misc SV
    "OwenSamplingSV": {"color": "#7DCE82", "marker": "o"},
    "StratifiedSamplingSV": {"color": "#4B7B4E", "marker": "o"},
    # Regression MSR
    "ProxySPEX": {"color": "#ef27a6", "marker": "o"},
    "ProxySHAP (Linear, MSR-b) [our]": {
        "color": "#15B01A",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP (Linear, MSR) [our]": {
        "color": "#15B01A",
        "marker": "o",
        "linestyle": "dashed",
    },
    "ProxySHAP (Linear) [our]": {
        "color": "#15B01A",
        "marker": "o",
        "linestyle": "dotted",
    },
    "ProxySHAP+ (XGBoost, MSR-b) [our]": {
        "color": "#1e25e5",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP+ (XGBoost, MSR) [our]": {
        "color": "#1e25e5",
        "marker": "o",
        "linestyle": "dashed",
    },
    "ProxySHAP+ (XGBoost) [our]": {
        "color": "#1e25e5",
        "marker": "o",
        "linestyle": "dotted",
    },
    "ProxySHAP* (XGBoost, MSR-b) [our]": {
        "color": "#06C2AC",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP* (XGBoost, MSR) [our]": {
        "color": "#06C2AC",
        "marker": "o",
        "linestyle": "dashed",
    },
    "ProxySHAP* (XGBoost) [our]": {
        "color": "#06C2AC",
        "marker": "o",
        "linestyle": "dotted",
    },
    "ProxySHAP (XGBoost) [our]": {
        "color": "#1e88e5",
        "marker": "o",
        "linestyle": "dotted",
    },
    "ProxySHAP (XGBoost, MSR-b) [our]": {
        "color": "#1e88e5",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP (XGBoost, MSR) [our]": {
        "color": "#1e88e5",
        "marker": "o",
        "linestyle": "dashed",
    },
    "ProxySHAP-Special [our]": {
        "color": "#e90000",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP-Special (Value) [our]": {
        "color": "#13e900",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP-Special (Marginal) [our]": {
        "color": "#00f2ff",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP-Special (Value, KernelSHAP) [our]": {
        "color": "#ff00e6",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP-Special (Simple, InverseBinom) [our]": {
        "color": "#692254",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP-Special (Simple, KernelSHAP) [our]": {
        "color": "#063606",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP-Special (Two-Stage) [our]": {
        "color": "#df5050",
        "marker": "o",
        "linestyle": "solid",
    }



}
STYLE_DICT = defaultdict(lambda: {"color": "black", "marker": "o"}, STYLE_DICT)
MARKERS = []
LIGHT_GRAY = "#d3d3d3"
LINE_STYLES_ORDER = {
    0: "solid",
    1: "solid",
    2: "solid",
    3: "dashed",
    4: "dashdot",
    "all": "solid",
}
APPROXIMATOR_TO_ZORDER = {
    ## Tree-base
    "ProxySHAP (Linear Det. MSR) [our]": 3,
    "ProxySHAP (Linear Prob. MSR) [our]": 3,
    "ProxySHAP (Linear) [our]": 3,
    "ProxySHAP (XGBoost Det. MSR) [our]": 5,
    "ProxySHAP (XGBoost Prob. MSR) [our]": 5,
    "ProxySHAP (XGBoost) [our]": 5,
    "ProxySHAP+ (XGBoost Det. MSR) [our]": 5,
    "ProxySHAP+ (XGBoost Prob. MSR) [our]": 5,
    "ProxySHAP+ (XGBoost) [our]": 5,
    "ProxySHAP* (XGBoost Det. MSR) [our]": 4,
    "ProxySHAP* (XGBoost Prob. MSR) [our]": 4,
    "ProxySHAP* (XGBoost) [our]": 4,
    "ProxySHAP (XGBoost FixedHPO Det. MSR) [our]": 3,
    "ProxySHAP (XGBoost FixedHPO Prob. MSR) [our]": 3,
    "ProxySHAP (DT) [our]": 7,
    "ProxySHAP-Special [our]": 7,
    "KernelSHAPIQ": 2,
    "ProxySpex": 1,
}
APPROXIMATOR_TO_ZORDER = defaultdict(lambda: 3, APPROXIMATOR_TO_ZORDER)
LINE_MARKERS_ORDER = {0: "o", 1: "o", 2: "o", 3: "X", 4: "d", "all": "o"}
LINE_THICKNESS = 2
MARKER_SIZE = 7


LOG_SCALE_MAX = 1e2
LOG_SCALE_MIN = 1e-10

METRICS_LIMITS = {
    "Precision@10": (0, 1),
    "Precision@5": (0, 1),
    "KendallTau": (-1, 1),
    "KendallTau@5": (-1, 1),
    "KendallTau@10": (-1, 1),
    "KendallTau@50": (-1, 1),
}
METRICS_NOT_TO_LOG_SCALE = list(METRICS_LIMITS.keys())


def create_application_name(setup: str, *, abbrev: bool = False) -> str:
    """Create an application name from the setup string."""
    application_name = "".join(setup.split("_")[0:2])
    application_name = application_name.replace("Game", "")
    application_name = application_name.replace("SynthData", "")
    application_name = application_name.replace("AdultCensus", "")
    application_name = application_name.replace("CaliforniaHousing", "")
    application_name = application_name.replace("BikeSharing", "")
    application_name = application_name.replace("ImageClassifier", "LocalExplanation")
    application_name = application_name.replace("SentimentAnalysis", "LocalExplanation")
    application_name = application_name.replace("TreeSHAPIQXAI", "LocalExplanation")
    application_name = application_name.replace(
        "RandomForestEnsembleSelection",
        "EnsembleSelection",
    )
    if abbrev:
        application_name = abbreviate_application_name(application_name)
    return application_name


def abbreviate_application_name(
    application_name: str, *, new_line: bool = False
) -> str:
    """Abbreviate the application name.

    Abbreviate the application name by taking the first three characters after each capital
    letter and adding a dot. The last character is not abbreviated.

    Args:
        application_name: The application name to abbreviate.
        new_line: Whether to add a new line after each abbreviation. Defaults to ``False``.

    Returns:
        The abbreviated application name.

    Example:
        >>> abbreviate_application_name("LocalExplanation")
        "Loc. Exp."

    """
    abbreviations = []
    count_char = 0
    for char in application_name:
        if char.isupper():
            count_char = 0
            abbreviations.append(char)
        else:
            count_char += 1
            if count_char == 3:
                abbreviations.append(".")
            elif count_char > 3:
                continue
            else:
                abbreviations.append(char)
    abbreviation = "".join(abbreviations)
    if application_name == "DatasetValuation":
        abbreviation = "Dst. Val."
    if application_name == "SOUM":
        abbreviation = "SOUM"
    if application_name == "SOUM (low)" and new_line:
        abbreviation = "SOUM\n(low)"
    if application_name == "SOUM (high)":
        abbreviation = "SOUM\n(high)"
    if new_line:
        abbreviation = abbreviation.replace(".", ".\n")
    return abbreviation.strip()


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


def agg_percentile(q: float) -> Callable[[np.ndarray], np.floating]:
    """Get the aggregation function for the given percentile.

    Args:
        q: The percentile to compute.

    Returns:
        The aggregation function.

    """

    def quantile(x: np.ndarray) -> np.floating:
        """Performs the aggregation function for the given percentile."""
        return np.percentile(x, q)

    quantile.__name__ = f"quantile_{q}"
    return quantile


def plot_approximation_quality_vstime(
    data: pd.DataFrame | None = None,
    *,
    data_path: str | None = None,
    metric: str = "MSE",
    orders: list[int | str] | None = None,
    approximators: list[str] | None = None,
    aggregation: str = "mean",
    confidence_metric: str | None = "sem",
    log_scale_y: bool = False,
    log_scale_min: float = LOG_SCALE_MIN,
    log_scale_max: float = LOG_SCALE_MAX,
    log_scale_x: bool = False,
    legend: bool = True,
    remove_spines: bool = False,
    figsize: tuple[float, float] = (5, 4),
    marker_size: float = MARKER_SIZE,
    linewidth: float = LINE_THICKNESS,
    highlight_size: float = 1.5,
    time_column: str = "total_runtime",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the approximation quality curves.

    Args:
        data: The data to plot the values from (if `None`, the data_path must be provided).
        data_path: The path to the data to plot the values from (if `None`, the data must be
            provided).
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
        log_scale_min: The minimum value for the log scale. Defaults to 1e-7.
        log_scale_max: The maximum value for the log scale. Defaults to 1e2.
        legend: Whether to add a legend to the plot. Defaults to `True`.
        remove_spines: Whether to remove the spines in the top and right of the plot. Defaults to
            `False`.

    Returns:
        The figure and axes of the plot.

    """
    if data_path is None and data is None:
        msg = "Either data or data_path must be provided."
        raise ValueError(msg)

    if data is None:
        data = pd.read_csv(data_path)
    # remove exact
    data = data[~data["approximator"].str.contains("Exact")]
    # data = data.fillna(0)
    # get the metric data
    metric_data = get_metric_data_time(data, metric, time_column=time_column)

    # sort by time
    metric_data = metric_data.sort_values(by=time_column)

    # make sure orders is a list
    if orders is None:
        orders = ["all"]
    if isinstance(orders, int | str):
        orders = [orders]

    # make sure approximators is a list
    if approximators is None:
        approximators = list(metric_data["approximator"].unique())

    # set the confidence metrics
    confidence_metric_low, confidence_metric_high = confidence_metric, confidence_metric
    if confidence_metric is not None and "quantile" in confidence_metric:
        confidence_metric_low = "quantile_5"
        confidence_metric_high = "quantile_95"

    # create the plot
    plt.style.use(["science", "no-latex"])
    fig, ax = plt.subplots(figsize=figsize)
    approx_max_x = 0
    for approximator in approximators:
        for order in metric_data["order"].unique():
            if orders is not None and order not in orders:
                continue
            data_order = metric_data[
                (metric_data["approximator"] == approximator)
                & (metric_data["order"] == order)
            ].copy()

            if log_scale_y:
                # manually set all below log_scale_min to log_scale_min (to avoid log(0))
                data_order[aggregation] = data_order[aggregation].apply(
                    lambda x: max(x, log_scale_min),
                )

            # get the plot colors and styles
            line_style, line_marker = (
                LINE_STYLES_ORDER[order],
                LINE_MARKERS_ORDER[order],
            )
            color = STYLE_DICT[approximator]["color"]
            line_marker = STYLE_DICT[approximator]["marker"]
            line_style = STYLE_DICT[approximator].get("linestyle", line_style)

            ax.plot(
                data_order[time_column],
                data_order[aggregation],
                color="white",
                linestyle=line_style,
                marker=line_marker,
                linewidth=linewidth + highlight_size,
                markersize=marker_size + highlight_size,
                # markersize=MARKER_SIZE,
                zorder=APPROXIMATOR_TO_ZORDER[approximator],
            )

            # plot the mean values
            ax.plot(
                data_order[time_column],
                data_order[aggregation],
                color=color,
                linestyle=line_style,
                marker=line_marker,
                linewidth=linewidth,
                markersize=marker_size,
                # markersize=MARKER_SIZE,
                zorder=APPROXIMATOR_TO_ZORDER[approximator],
            )
            # plot the error bars if the confidence metric is not None
            if confidence_metric is not None:
                ax.fill_between(
                    data_order[time_column],
                    data_order[aggregation] - data_order[confidence_metric_low],
                    data_order[aggregation] + data_order[confidence_metric_high],
                    alpha=0.4,
                    color=color,
                )

    # add x/y labels
    ax.set_ylabel(metric)
    ax.set_xlabel(r"Total Runtime (s)" if time_column == "total_runtime" else time_column)

    # add grid to x-axis
    # ax.grid(axis="x", color=LIGHT_GRAY, linestyle="dashed")

    # add the legend
    if legend:
        add_legend(ax, approximators, orders=orders)
    # Log-scale x-axis
    ax.set_xscale("log")

    # set the y-axis limits
    if log_scale_y and metric not in METRICS_NOT_TO_LOG_SCALE:
        _set_y_axis_log_scale(ax, log_scale_min, log_scale_max)

    # set the y-axis limits for specific metrics
    if metric in METRICS_LIMITS:
        ax.set_ylim(METRICS_LIMITS[metric])

    if remove_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # resize the figure and remove padding
    plt.tight_layout()
    # raise ValueError("Either data or data_path must be provided.")
    return fig, ax


def plot_approximation_quality(
    data: pd.DataFrame | None = None,
    *,
    data_path: str | None = None,
    metric: str = "MSE",
    orders: list[int | str] | None = None,
    approximators: list[str] | None = None,
    aggregation: str = "mean",
    confidence_metric: str | None = "sem",
    log_scale_y: bool = False,
    log_scale_min: float = LOG_SCALE_MIN,
    log_scale_max: float = LOG_SCALE_MAX,
    log_scale_x: bool = False,
    legend: bool = True,
    remove_spines: bool = False,
    figsize: tuple[float, float] = (5, 4),
    marker_size: float = MARKER_SIZE,
    linewidth: float = LINE_THICKNESS,
    highlight_size: float = 1.5,
    style_dict: dict[str, dict[str, str]] | None = None,
    plot_labels: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the approximation quality curves.

    Args:
        data: The data to plot the values from (if `None`, the data_path must be provided).
        data_path: The path to the data to plot the values from (if `None`, the data must be
            provided).
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
        log_scale_min: The minimum value for the log scale. Defaults to 1e-7.
        log_scale_max: The maximum value for the log scale. Defaults to 1e2.
        legend: Whether to add a legend to the plot. Defaults to `True`.
        remove_spines: Whether to remove the spines in the top and right of the plot. Defaults to
            `False`.

    Returns:
        The figure and axes of the plot.

    """
    if style_dict is not None:
        global STYLE_DICT
        STYLE_DICT = style_dict
    if data_path is None and data is None:
        msg = "Either data or data_path must be provided."
        raise ValueError(msg)

    if data is None:
        data = pd.read_csv(data_path)
    # remove exact
    data = data[~data["approximator"].str.contains("Exact")]
    # data = data.fillna(0)
    # get the metric data
    metric_data = get_metric_data(data, metric)

    sorted_budget = list(data["budget"].sort_values(ascending=False).unique())

    # try:
    #     y_lim_min_budget = sorted_budget[3] if sorted_budget[0] >= 2**17 else sorted_budget[2]
    # except IndexError:
    #     y_lim_min_budget = sorted_budget[0]
    y_lim_min_budget = sorted_budget[0]
    # get min metric_value for y_lim

    min_value_y = (
        data[data["budget"] == y_lim_min_budget][metric].min() + 1e-7
    )  # Enforce >0 to have sensible log scale. Otherwise the bot_lim calculation below fails.
    # round value down to next decimal
    bot_lim = f"{min_value_y:.2e}"  # get the top limit in scientific notation
    # print(f"min_value_y: {min_value_y}, bot_lim: {bot_lim}, split: {bot_lim.split('e')}")
    bot_lim = bot_lim.split("e")[1]  # get the exponent
    bot_lim = int(bot_lim)  # get the top limit as the exponent + 1
    bot_lim = 10**bot_lim  # get the top limit in scientific notation
    #log_scale_min = max(log_scale_min, bot_lim)
    # make sure orders is a list
    if orders is None:
        orders = ["all"]
    if isinstance(orders, int | str):
        orders = [orders]

    # make sure approximators is a list
    if approximators is None:
        approximators = list(metric_data["approximator"].unique())

    # set the confidence metrics
    confidence_metric_low, confidence_metric_high = confidence_metric, confidence_metric
    if confidence_metric is not None and "quantile" in confidence_metric:
        confidence_metric_low = "quantile_5"
        confidence_metric_high = "quantile_95"

    # create the plot
    plt.style.use(["science", "no-latex"])
    fig, ax = plt.subplots(figsize=figsize)
    approx_max_x = 0
    min_positive_budget: float | None = None
    plotted_budgets: list[float] = []
    for approximator in approximators:
        for order in metric_data["order"].unique():
            if orders is not None and order not in orders:
                continue
            data_order = metric_data[
                (metric_data["approximator"] == approximator)
                & (metric_data["order"] == order)
            ].copy()

            # Log-scale x cannot display 0 or negative budgets.
            if log_scale_x:
                data_order = data_order[data_order["used_budget"] > 0].copy()
                if data_order.empty:
                    continue
                current_min = float(data_order["used_budget"].min())
                min_positive_budget = (
                    current_min
                    if min_positive_budget is None
                    else min(min_positive_budget, current_min)
                )

            # Track plotted x values so we can set limits/ticks from the actual data
            # (Matplotlib's default log locator often starts at 1 and skips endpoints).
            if not data_order.empty:
                plotted_budgets.extend(map(float, data_order["used_budget"].unique()))

            if log_scale_y:
                # manually set all below log_scale_min to log_scale_min (to avoid log(0))
                data_order[aggregation] = data_order[aggregation].apply(
                    lambda x: max(x, log_scale_min),
                )
            # get the plot colors and styles
            line_style, line_marker = (
                LINE_STYLES_ORDER[order],
                LINE_MARKERS_ORDER[order],
            )
            color = STYLE_DICT[approximator]["color"]
            line_marker = STYLE_DICT[approximator]["marker"]
            line_style = STYLE_DICT[approximator].get("linestyle", line_style)
            linewidth = STYLE_DICT[approximator].get("linewidth", linewidth)
            zorder = STYLE_DICT[approximator].get(
                "zorder", APPROXIMATOR_TO_ZORDER[approximator]
            )
            marker_size = STYLE_DICT[approximator].get("marker_size", marker_size)
            highlight_size = STYLE_DICT[approximator].get(
                "highlight_size",
                highlight_size,
            )

            ax.plot(
                data_order["used_budget"],
                data_order[aggregation],
                color="white",
                linestyle=line_style,
                marker=line_marker,
                linewidth=linewidth + highlight_size,
                markersize=marker_size + highlight_size,
                # markersize=MARKER_SIZE,
                zorder=zorder,
            )

            # plot the mean values
            ax.plot(
                data_order["used_budget"],
                data_order[aggregation],
                color=color,
                linestyle=line_style,
                marker=line_marker,
                linewidth=linewidth,
                markersize=marker_size,
                # markersize=MARKER_SIZE,
                zorder=zorder,
            )
            # plot the error bars if the confidence metric is not None
            if confidence_metric is not None:
                ax.fill_between(
                    data_order["used_budget"],
                    data_order[aggregation] - data_order[confidence_metric_low],
                    data_order[aggregation] + data_order[confidence_metric_high],
                    alpha=0.4,
                    color=color,
                )
                # ax.errorbar(
                #     data_order["used_budget"],
                #     data_order[aggregation],
                #     yerr=[
                #         data_order[confidence_metric_low],
                #         data_order[confidence_metric_high],
                #     ],
                #     fmt="none",
                #     ecolor=color,
                #     capsize=3,
                #     alpha=0.8,
                #     zorder=APPROXIMATOR_TO_ZORDER[approximator],
                # )
            approx_max_x = max(approx_max_x, int(data_order["used_budget"].max()))

    # add x/y labels
    if plot_labels:
        ax.set_ylabel(metric)
        ax.set_xlabel(r"Model Evaluations (relative to $2^n$)")

    # add grid to x-axis
    ax.grid(axis="x", color=LIGHT_GRAY, linestyle="dashed")

    # add the legend
    if legend:
        add_legend(ax, approximators, orders=orders)

    # Log-scale x-axis
    if log_scale_x:
        # set the x-axis to a log scale
        ax.set_xscale("log")
        # keep view focused on the actually plotted positive budgets
        # if plotted_budgets:
        #     positive_budgets = [b for b in plotted_budgets if b > 0]
        #     if positive_budgets:
        #         ax.set_xlim(
        #             left=min(positive_budgets) * 0.9, right=max(positive_budgets) * 1.1
        #         )
        # elif min_positive_budget is not None:
        #     ax.set_xlim(left=min_positive_budget, right=approx_max_x)
    ax.set_xscale("log")

    ## Set y-axis log scale ##

    # set the y-axis limits
    if log_scale_y and metric not in METRICS_NOT_TO_LOG_SCALE:
        _set_y_axis_log_scale(ax, log_scale_min, log_scale_max)

    # set the y-axis limits for specific metrics
    if metric in METRICS_LIMITS:
        ax.set_ylim(METRICS_LIMITS[metric])

    # add %model calls to the x-axis as a secondary axis
    # _set_x_axis_ticks(
    #     ax,
    #     n_players=int(data["n_players"].unique().max()),
    #     max_budget=approx_max_x,
    #     log_scale_x=log_scale_x,
    #     plotted_budgets=plotted_budgets,
    # )

    if remove_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # resize the figure and remove padding
    plt.tight_layout()
    # raise ValueError("Either data or data_path must be provided.")
    return fig, ax


def _set_x_axis_ticks(
    ax: plt.Axes,
    *,
    n_players: int,
    max_budget: int,
    log_scale_x: bool = False,
    plotted_budgets: list[float] | None = None,
) -> None:
    """Sets the x-axis ticks.

    For linear scaling and small n, uses 25% intervals. For log scaling (or large n),
    derives ticks from Matplotlib's tick locator and formats them with absolute
    budgets and relative percentages.
    """
    # If log-scale x is requested, derive ticks from the actual plotted budgets.
    if log_scale_x and plotted_budgets:
        unique_budgets = np.array(sorted({b for b in plotted_budgets if b > 0}))
        if unique_budgets.size == 0:
            return

        # Choose a small number of representative ticks (log-spaced) and snap them
        # to the nearest real budgets so min/max match the dataset exactly.
        n_ticks = int(min(6, unique_budgets.size))
        targets = np.geomspace(unique_budgets[0], unique_budgets[-1], num=n_ticks)
        snapped: list[float] = []
        for t in targets:
            idx = int(np.searchsorted(unique_budgets, t, side="left"))
            if idx <= 0:
                snapped.append(float(unique_budgets[0]))
            elif idx >= unique_budgets.size:
                snapped.append(float(unique_budgets[-1]))
            else:
                left = unique_budgets[idx - 1]
                right = unique_budgets[idx]
                snapped.append(float(left if (t - left) <= (right - t) else right))

        budgets = np.array(
            sorted(set(snapped) | {float(unique_budgets[0]), float(unique_budgets[-1])})
        )
        budgets_relative = (
            budgets / (2**n_players)
            if n_players < 64
            else np.zeros_like(budgets) + 0.01
        )

    elif (n_players <= 16) and (not log_scale_x):
        # only for small number of players set the ticks as 25% intervals
        budgets_relative = np.arange(0, 1.25, 0.25)
        budgets = budgets_relative * (2**n_players)
    else:
        budgets = ax.get_xticks()
        # remove negative values; remove 0 for log scale
        budgets = budgets[budgets > 0] if log_scale_x else budgets[budgets >= 0]
        # remove all values greater than max_budget * 1.05
        budgets = budgets[budgets <= max_budget * 1.05]
        budgets_relative = (
            budgets / (2**n_players)
            if n_players < 64
            else np.zeros_like(budgets) + 0.01
        )

    xtick_labels = []
    for bdgt, bdgt_rel in zip(budgets, budgets_relative, strict=False):
        bdgt_rel_str = f"{bdgt_rel:.0%}"
        if bdgt_rel <= 0.01 and bdgt_rel != 0:
            bdgt_rel_str = "1%"
        if bdgt_rel == 0:
            xtick_labels.append("0")
        else:
            xtick_labels.append(f"{int(bdgt)}\n({bdgt_rel_str})")

    ax.set_xticks(budgets)
    ax.set_xticklabels(xtick_labels)


def _set_y_axis_log_scale(
    ax: plt.Axes, log_scale_min: float, log_scale_max: float
) -> None:
    """Sets the y-axis to a log scale and adjusts the limits."""
    # adjust the top limit to be one order of magnitude higher than the current top limit
    top_lim = ax.get_ylim()[1]
    top_lim = f"{top_lim:.2e}"  # get the top limi in scientific notation
    top_lim = top_lim.split("e")[1]  # get the exponent
    top_lim = int(top_lim) + 1  # get the top limit as the exponent + 1
    top_lim = 10**top_lim  # get the top limit in scientific notation
    top_lim = min(top_lim, log_scale_max)

    # Set log scaling and explicit limits.
    ax.set_yscale("log")
    ax.set_ylim(bottom=log_scale_min * 1.1, top=top_lim)

    ## Ensure minor ticks are shown on log scale ##
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


def get_metric_data_time(
    results_df: pd.DataFrame,
    metric: str = "MSE",
    time_column: str = "total_runtime",
) -> pd.DataFrame:
    """Get the metric data for the given results.

    Args:
        results_df: The results dataframe.
        metric: The metric to get the data for. Defaults to "MSE".
        time_column: The time column to use for grouping. Defaults to "total_runtime".

    Returns:
        The metric data.

    """
    # get the metric columns for each order in the results
    metric_columns = [col for col in results_df.columns if metric == col]
    print(f"Metric columns: {metric_columns}: Metric: {metric}")
    metric_dfs = []
    for metric_col in metric_columns:
        data_order = (
            results_df.groupby(["approximator", "iteration", time_column])
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
                    ],
                },
            )
            .reset_index()
        )
        data_order["order"] = (
            "all" if "_" not in metric_col else int(metric_col.split("_")[0])
        )
        # rename the columns of grouped data
        new_columns = [
            "_".join(col).strip() if col[1] != "" else col[0]
            for col in data_order.columns
        ]
        new_columns = [col.replace(f"{metric_col}_", "") for col in new_columns]

        data_order.columns = new_columns
        metric_dfs.append(data_order)

    # concat the dataframes along the row
    metric_df = pd.concat(metric_dfs)

    # compute the standard error
    metric_df["sem"] = (
        metric_df["std"] / metric_df["count"] ** 0.5
    )  # compute standard error

    return metric_df


def get_metric_data(results_df: pd.DataFrame, metric: str = "MSE") -> pd.DataFrame:
    """Get the metric data for the given results.

    Args:
        results_df: The results dataframe.
        metric: The metric to get the data for. Defaults to "MSE".

    Returns:
        The metric data.

    """
    # get the metric columns for each order in the results
    metric_columns = [col for col in results_df.columns if metric == col]
    print(f"Metric columns: {metric_columns}: Metric: {metric}")
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
                    ],
                },
            )
            .reset_index()
        )
        data_order["order"] = (
            "all" if "_" not in metric_col else int(metric_col.split("_")[0])
        )
        # rename the columns of grouped data
        new_columns = [
            "_".join(col).strip() if col[1] != "" else col[0]
            for col in data_order.columns
        ]
        new_columns = [col.replace(f"{metric_col}_", "") for col in new_columns]

        data_order.columns = new_columns
        metric_dfs.append(data_order)

    # concat the dataframes along the row
    metric_df = pd.concat(metric_dfs)

    # compute the standard error
    metric_df["sem"] = (
        metric_df["std"] / metric_df["count"] ** 0.5
    )  # compute standard error

    return metric_df


def add_legend(
    axis: plt.Axes,
    approximators: list[str | Approximator],
    *,
    orders: list[int | str] | None = None,
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

    # convert approximators to strings if they are not strings
    if approximators is not None:
        approximators_str = []
        for approx in approximators:
            approx_str = type(approx).__name__
            if approx_str == "str":
                approx_str = approx
            approximators_str.append(approx_str)
        approximators = approximators_str

    # plot the order elements
    if orders is not None:
        if isinstance(orders, int | str):
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
            linestyle=STYLE_DICT[approximator].get(
                "linestyle",
                "solid",
            ),
            linewidth=LINE_THICKNESS,
        )
    leg = axis.legend(
        loc=loc, bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, framealpha=1
    )
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_facecolor("none")
