from typing import Union

import pandas as pd
import matplotlib.pyplot as plt

from shapiq.approximator._base import Approximator

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

LINE_STYLES_ORDER = {0: "solid", 1: "dotted", 2: "solid", 3: "dashed", 4: "dashdot"}


def get_color(approximator: Union[str, Approximator]) -> str:
    """Get the color for the given approximator.

    Args:
        approximator: The approximator to get the color for.

    Returns:
        The color for the approximator.
    """
    if isinstance(approximator, Approximator):
        approximator = approximator.__class__.__name__

    return COLORS[approximator]


def plot_curves(metric_values: pd.DataFrame, metric: str = "MSE") -> tuple[plt.Figure, plt.Axes]:
    """Plot the approximation quality curves.

    Args:
        metric_values: The metric values to plot.
        metric: The metric to plot. Defaults to `"MSE"`.

    Returns:
        The figure and axes of the plot.
    """
    fig, ax = plt.subplots()

    for approximator, data in metric_values.groupby("approximator"):
        data = data.reset_index()
        for order, data_order in data.groupby("order"):
            line_style = LINE_STYLES_ORDER[int(order)]
            ax.plot(
                data_order["budget"],
                data_order[metric]["mean"],
                label=approximator,
                color=get_color(str(approximator)),
                linestyle=line_style,
            )
            # plot the error bars
            ax.fill_between(
                data_order["budget"],
                data_order[metric]["mean"] - data_order[metric]["std"],
                data_order[metric]["mean"] + data_order[metric]["std"],
                alpha=0.0,
                color=get_color(str(approximator)),
            )

    return fig, ax


if __name__ == "__main__":

    GAME_NAME = "Language Model"

    data = pd.read_csv("results.csv")
    metric = "MSE"  # "MSE"

    metric_data = (
        data.groupby(["approximator", "order", "budget"])
        .agg({metric: ["mean", "std"]})
        .reset_index()
    )

    params = {
        "legend.fontsize": "x-large",
        "figure.figsize": (6, 7),
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    plt.rcParams.update(params)
    fig, ax = plot_curves(metric_data, metric=metric)

    ax.set_ylim(0, 0.01)
    ax.set_xlabel("Budget")
    ax.set_ylabel(metric)
    ax.set_title(GAME_NAME)

    plt.show()
