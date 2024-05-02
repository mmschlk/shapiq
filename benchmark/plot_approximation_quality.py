from typing import Union, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from shapiq.approximator._base import Approximator

COLORS = {
    # permutation sampling
    "PermutationSamplingSII": "#7d53de",
    "PermutationSamplingSTII": "#7d53de",
    "PermutationSamplingSV": "#7d53de",
    # KernelSHAP-IQ
    "KernelSHAPIQ": "#ff6f00",
    "InconsistentKernelSHAPIQ": "#ffba08",
    # SVARM-based
    "SVARMIQ": "#00b4d8",
    "SVARM": "#00b4d8",
    # shapiq
    "SHAPIQ": "#ef27a6",
}

LINE_STYLES_ORDER = {0: "solid", 1: "dotted", 2: "solid", 3: "dashed", 4: "dashdot", "all": "solid"}
LINE_MARKERS_ORDER = {0: "o", 1: "o", 2: "s", 3: "X", 4: "d", "all": "o"}
LINE_THICKNESS = 1


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
    metric_values: pd.DataFrame,
    orders: list[Union[int, str]] = None,
    approximators: list[str] = None,
    aggregation: str = "mean",
    confidence_metric: Optional[str] = "sem",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the approximation quality curves.

    Args:
        metric_values: The metric values to plot.
        orders: The orders to plot. If `None`, all orders are plotted. Defaults to `None`.
            Can be a list of integers or a single integer.
        approximators: The approximators to plot. Defaults to `None`. When `None`, all approximators
            are plotted.
        aggregation: The aggregation function to plot for the metric values. Defaults to "mean".
            Available options are "mean", "median", "quantile_95", "quantile_5".
        confidence_metric: The metric to use for the confidence interval. Defaults to "sem".

    Returns:
        The figure and axes of the plot.
    """
    # make sure orders is a list
    if orders is None:
        orders = "all"
    if isinstance(orders, int) or isinstance(orders, str):
        orders = [orders]

    # make sure approximators is a list
    if approximators is None:
        approximators = list(metric_values["approximator"].unique())

    # set the confidence metrics
    confidence_metric_low, confidence_metric_high = confidence_metric, confidence_metric
    if confidence_metric is not None and "quantile" in confidence_metric:
        confidence_metric_low = "quantile_5"
        confidence_metric_high = "quantile_95"

    # create the plot
    fig, ax = plt.subplots()
    for approximator in approximators:
        for order in metric_values["order"].unique():
            if orders is not None and order not in orders:
                continue
            data_order = metric_values[
                (metric_values["approximator"] == approximator) & (metric_values["order"] == order)
            ]
            # get the plot colors and styles
            line_style, line_marker = LINE_STYLES_ORDER[order], LINE_MARKERS_ORDER[order]
            color = COLORS.get(approximator, "black")

            # plot the mean values
            ax.plot(
                data_order["budget"],
                data_order[aggregation],
                color=color,
                linestyle=line_style,
                marker=line_marker,
                linewidth=LINE_THICKNESS,
                mec="white",
            )
            # plot the error bars if the confidence metric is not None
            if confidence_metric is not None:
                ax.fill_between(
                    data_order["budget"],
                    data_order[aggregation] - data_order[confidence_metric_low],
                    data_order[aggregation] + data_order[confidence_metric_high],
                    alpha=0.1,
                    color=color,
                )

    return fig, ax


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
            data.groupby(["approximator", "budget", "iteration"])
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
    orders: list[Union[int, str]] = None,
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
            color=COLORS.get(approximator, "black"),
            linewidth=LINE_THICKNESS,
        )

    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc=loc)


if __name__ == "__main__":

    data_path = "results_test.csv"
    save_path = "plot.pdf"

    # parameters for the plot  ---------------------------------------------------------------------
    # title of the plot
    title = "Language Model"
    # a list of approximators to plot (in that order)
    approximators_to_plot = [
        "KernelSHAPIQ",
        "InconsistentKernelSHAPIQ",
        "SVARMIQ",
        "SHAPIQ",
        "PermutationSamplingSII",
    ]
    # a list of orders to plot (in that order)
    orders_to_plot = [2]  # can be "all" or 1, 2, 3, etc.
    # the metric to plot
    metric = "MSE"  # can be "MSE", "MAE", "Precision@10", "Precision@5"
    # denotes weather to plot the legend or not
    plot_legend = True
    # ylim of the plot
    ylim = None  # (0, 3.65e-3)  # can be None or a tuple of (min, max) or None
    # make y-axis log scale
    log_scale = True
    scientific_notation = False
    # matplotlib parameters
    params = {
        "legend.fontsize": "x-large",
        "figure.figsize": (6, 7),
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    plt.rcParams.update(params)

    # make the plot --------------------------------------------------------------------------------

    data = pd.read_csv(data_path)
    metric_data = get_metric_data(data, metric)  # get the metric data from the results file

    fig, ax = plot_approximation_quality(
        metric_data, approximators=approximators_to_plot, orders=orders_to_plot
    )

    # add the legend
    if plot_legend:
        add_legend(ax, approximators=approximators_to_plot, orders=[2])

    ax.set_xlabel("Budget")
    ax.set_ylabel(metric)

    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(metric + " (log scale)")
    elif scientific_notation:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        ax.set_ylim(ylim)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
