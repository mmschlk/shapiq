import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapiq.benchmark.plot import (
    _set_x_axis_ticks,
    STYLE_DICT,
    LINE_THICKNESS,
    MARKER_SIZE,
)

from plot_approximation import (
    TITLE_FONT_SIZE,
    APPROXIMATOR_RENAMING,
    parse_approximator,
    compute_value,
)

DATA_NAMES = {
    "breast_cancer": "Cancer ($d=30$)",
    "communities_and_crime": "Crime ($d=101$)",
    "corrgroups60": "CG60 ($d=60$)",
    "forest_fires": "Forest ($d=13$)",
    "independentlinear60": "IL60 ($d=60$)",
    "nhanesi": "NHANES ($d=79$)",
    "real_estate": "Estate ($d=15$)",
    "wine_quality": "Wine ($d=11$)",
    "adult_census": "Adult ($d=14$)",
    "california_housing": "Housing ($d=8$)",
    "bike_sharing": "Bike ($d=12$)",
    "ViT4by4Patches": "ViT16 ($d=16$)",
    "ViT3by3Patches": "ViT9 ($d=9$)",
    "ResNet18w14Superpixel": "ResNet18 ($d=14$)",
    "SentimentIMDBDistilBERT14": "DistilBERT ($d=14$)",
    "CIFAR10": "CIFAR10-ViT16 ($d=16$)",
}


LINE_STYLES_COMPONENT = {"Runtime": "-", "Evaluations": ":", "Computation": "-", "Linear Fit": "--"}


def plot_runtime(group_sorted, plot_legend=False):
    # create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    for approximator in approximators_to_plot:
        color = STYLE_DICT[approximator]["color"]

        subset = group_sorted[group_sorted["approximator"] == approximator]

        if len(subset) > 0:
            total_linear_fit = np.polyfit(subset["budget"], subset["total_mean"], 1)
            subset["total_linear_fit"] = total_linear_fit[0] * subset["budget"] + total_linear_fit[1]
            regression_linear_fit = np.polyfit(subset["budget"], subset["regression_mean"], 1)
            subset["regression_linear_fit"] = regression_linear_fit[0] * subset["budget"] + regression_linear_fit[1]
            evaluations_linear_fit = np.polyfit(
                subset["budget"], subset["evaluations_mean"], 1
            )
            subset["evaluations_linear_fit"] = (
                evaluations_linear_fit[0] * subset["budget"] + evaluations_linear_fit[1]
            )
            ax.plot(
                subset["budget"],
                subset["total_linear_fit"],
                linestyle="--",
                linewidth=LINE_THICKNESS,
                marker=None,
                color=color,
                alpha=0.3,
            )
            # ax.plot(
            #     subset["budget"],
            #     subset["regression_linear_fit"],
            #     linestyle="--",
            #     linewidth=LINE_THICKNESS,
            #     marker=None,
            #     color=color,
            #     alpha=0.3,
            # )
            # ax.plot(
            #     subset["budget"],
            #     subset["evaluations_linear_fit"],
            #     linestyle="--",
            #     linewidth=LINE_THICKNESS,
            #     color=color,
            #     marker=None,
            #     alpha=0.3,
            # )
        ax.plot(
            subset["budget"],
            subset["total_mean"],
            linestyle="-",
            marker="o",
            linewidth=LINE_THICKNESS,
            markersize=MARKER_SIZE,
            color=color,
        )
        ax.fill_between(
            subset["budget"],
            subset["total_mean"] - subset["total_sem"],
            subset["total_mean"] + subset["total_sem"],
            alpha=0.1,
            color=color,
        )
        # ax.plot(
        #     subset["budget"],
        #     subset["regression_mean"],
        #     linestyle="-",
        #     marker="o",
        #     linewidth=LINE_THICKNESS,
        #     markersize=MARKER_SIZE,
        #     color=color,
        # )
        # ax.fill_between(
        #     subset["budget"],
        #     subset["regression_mean"] - subset["regression_sem"],
        #     subset["regression_mean"] + subset["regression_sem"],
        #     alpha=0.1,
        #     color=color,
        # )
        # ax.plot(
        #     subset["budget"],
        #     subset["evaluations_mean"],
        #     linestyle=":",
        #     marker="o",
        #     linewidth=LINE_THICKNESS,
        #     markersize=MARKER_SIZE,
        #     color=color,
        # )
        # ax.fill_between(
        #     subset["budget"],
        #     subset["evaluations_mean"] - subset["evaluations_sem"],
        #     subset["evaluations_mean"] + subset["evaluations_sem"],
        #     alpha=0.1,
        #     color=color,
        # )

    if plot_legend:
        ax.plot([], [], label="$\\bf{PolySHAP}$", color="none")
        for component in ["Runtime","Linear Fit"]:
            if component in ["Runtime"]:
                marker="o"
            else:
                marker=None
            ax.plot(
                [],
                [],
                label=f"{component}",
                color="black",
                linestyle=LINE_STYLES_COMPONENT[component],
                marker=marker,
                linewidth=LINE_THICKNESS,
                mec="white",
            )
        ax.plot([], [], label="$\\bf{Method}$", color="none")
        for approximator in approximators_to_plot:
            ax.plot(
                [],
                [],
                label=approximator,
                color=STYLE_DICT[approximator]["color"],
                linewidth=LINE_THICKNESS,
            )
        ax.legend()
    n_players = group_sorted["n_players"].iloc[0]
    ax.set_yscale("log")
    ax.set_xlabel("Budget")
    _set_x_axis_ticks(
        ax, n_players, 20000
    )  # updated to pass ax if your function supports it
    ax.set_ylabel("Time (seconds)", fontsize=20)
    ax.set_xlabel(r"Budget ($m$)", fontsize=20)
    plt.yticks(fontsize=14)
    dataset = group_sorted["game_id"].unique()[0]
    # take the part before the last two underscores (_random_forest)
    if len(dataset.split("_")) > 2:
        dataset = "_".join(dataset.split("_")[:-2])
    else:
        dataset = dataset.split("_")[0]
    ax.set_title(
        f"{DATA_NAMES[dataset]}",
        fontsize=TITLE_FONT_SIZE,
    )

    return fig, ax


if __name__ == "__main__":
    # This script plots the runtime analysis results from the CSV file
    df = pd.read_csv("runtime_analysis.csv")
    df_cifar10 = pd.read_csv("runtime_analysis_cifar10.csv")
    df = pd.concat([df, df_cifar10], ignore_index=True)

    df = df[
        df["game_id"].isin(
            [
                "california_housing_random_forest",
                "breast_cancer_random_forest",
                "real_estate_random_forest",
                "independentlinear60_random_forest",
                "CIFAR10_1",
            ]
        )
    ]

    # aggregate runtimes by mean for each game_id, approximator
    runtime_results = (
        df.groupby(
            [
                "game_id",
                "approximator",
                "budget",
                "n_players",
                "id_config_approximator",
            ]
        )
        .agg(
            {
                col: ["mean", "std", "var", "count"]
                for col in [
                    "sampling",
                    "evaluations",
                    "regression",
                    "shapiq_post_processing",
                    "total",
                ]
            }
        )
        .reset_index()
    )
    runtime_results["regression_sem"] = runtime_results[
        ("regression", "std")
    ] / np.sqrt(runtime_results[("regression", "count")])
    runtime_results["evaluations_sem"] = runtime_results[
        ("evaluations", "std")
    ] / np.sqrt(runtime_results[("evaluations", "count")])
    runtime_results["total_sem"] = runtime_results[
        ("total", "std")
    ] / np.sqrt(runtime_results[("total", "count")])
    # flatten column multi-index
    runtime_results.columns = [
        "_".join(filter(None, col)).rstrip("_") for col in runtime_results.columns
    ]

    # columns to stack
    stack_cols = [
        "sampling_mean",
        "evaluations_mean",
        "regression_mean",
        "shapiq_post_processing_mean",
    ]

    # approximators to show
    approximators_to_plot = [
        "RegressionMSR",
        "LeverageSHAP",
        "PolySHAP-2ADD",
        "PolySHAP-3ADD",
        "PolySHAP-4ADD",
    ]

    runtime_results = runtime_results[
        runtime_results["approximator"].isin(approximators_to_plot)
    ]

    # plot legend
    fig, ax = plot_runtime(runtime_results, plot_legend=True)
    ax.axis("off")
    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    # Replace old labels with new ones
    labels = [APPROXIMATOR_RENAMING.get(l, l) for l in labels]
    # Update legend
    ax.legend(handles, labels)
    # Save the legend separately
    fig.savefig(f"plots/legend_runtime.pdf", bbox_inches="tight")
    # fig_legend.show()

    # filter based on budget threshold
    runtime_results[["p", "q"]] = runtime_results["approximator"].apply(
        lambda x: pd.Series(parse_approximator(x))
    )
    runtime_results["minimum_budget_to_plot"] = runtime_results.apply(
        compute_value, axis=1
    )
    runtime_results = runtime_results[
        runtime_results["budget"] >= runtime_results["minimum_budget_to_plot"]
    ]

    for game_id, group in runtime_results.groupby("game_id"):
        # sort by budget
        group_sorted = group.sort_values("budget")

        fig, ax = plot_runtime(group_sorted)
        # ax.legend()
        fig.tight_layout()
        fig.savefig("plots/runtime_analysis/runtime_analysis_" + game_id + ".pdf")
