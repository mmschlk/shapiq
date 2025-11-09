from __future__ import annotations

import pandas as pd
import math, re
from shapiq.benchmark import plot_approximation_quality
import numpy as np

import matplotlib.pyplot as plt

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
}

APPROXIMATOR_RENAMING = {
    "PermutationSampling": "Permutation Sampling",
    "KernelSHAP": "1-PolySHAP (KernelSHAP) ",
    "LeverageSHAP": "1-PolySHAP (KernelSHAP)",
    "PolySHAP-2ADD": "2-PolySHAP",
    "PolySHAP-3ADD": "3-PolySHAP",
    "PolySHAP-4ADD": "4-PolySHAP",
    "PolySHAP-2ADD-10%": "2-PolySHAP (10%)",
    "PolySHAP-2ADD-20%": "2-PolySHAP (20%)",
    "PolySHAP-2ADD-50%": "2-PolySHAP (50%)",
    "PolySHAP-2ADD-75%": "2-PolySHAP (75%)",
    "PolySHAP-3ADD-10%": "3-PolySHAP (10%)",
    "PolySHAP-3ADD-20%": "3-PolySHAP (20%)",
    "PolySHAP-3ADD-50%": "3-PolySHAP (50%)",
    "PolySHAP-3ADD-75%": "3-PolySHAP (75%)",
}

TITLE_FONT_SIZE = 24


# Function to extract p and q
def parse_approximator(s):
    if pd.isna(s):
        return np.nan, np.nan
    match = re.search(r"PolySHAP-(\d+)ADD(?:-(\d+)%)*", s)
    if not match:
        return np.nan, np.nan
    if match:
        p = int(match.group(1))
        q = int(match.group(2)) if match.group(2) else 100
        return p, q
    return None, None


def compute_value(row):
    p = row["p"]
    q = row["q"]
    n = int(row["n_players"])

    # treat NaN (or any missing) as "not a PolySHAP" -> return 0
    if pd.isna(p) or pd.isna(q):
        return 0.0

    # now safe to cast p,q to int
    p = int(p)
    q = int(q)

    # guard if p > n: combinations for k>n are zero, so sum_{i=1}^{p-1} reduces to i=1..n
    if p > n:
        return float(sum(math.comb(n, i) for i in range(1, n + 1)))

    total = sum(math.comb(n, i) for i in range(1, p))
    total += math.comb(n, p) * (q / 100.0)
    return float(total)


if __name__ == "__main__":
    # Load the results from the CSV file
    results_df = pd.read_csv("results_benchmark.csv")
    results_df = results_df.sort_values(by="n_players")

    GAME_IDS = results_df["game_id"].unique()
    GAME_TYPES = results_df["game_type"].unique()

    info = results_df[["game_id", "n_players"]].drop_duplicates()

    results_df[["p", "q"]] = results_df["approximator"].apply(
        lambda x: pd.Series(parse_approximator(x))
    )
    results_df["minimum_budget_to_plot"] = results_df.apply(compute_value, axis=1)
    results_df = results_df[
        results_df["used_budget"] >= results_df["minimum_budget_to_plot"]
    ]

    # Create and save a legend for the plots
    fig, ax = plot_approximation_quality(
        data=results_df,
        metric="MSE",
        log_scale_y=True,
        log_scale_x=False,
        legend=True,
    )

    ax.axis("off")
    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    # Replace old labels with new ones
    labels = [APPROXIMATOR_RENAMING.get(l, l) for l in labels]
    # Update legend
    ax.legend(handles, labels)
    # Save the legend separately
    fig.savefig(f"plots/legend.pdf", bbox_inches="tight")
    # fig_legend.show()

    # Plot approximation quality for standard
    plot_df = results_df[
        # (results_df["approximator"] == "MSR")
        # | (results_df["approximator"] == "SVARM")
        (results_df["approximator"] == "RegressionMSR")
        | (results_df["approximator"] == "PermutationSampling")
        # | (results_df["approximator"] == "KernelSHAP")
        | (results_df["approximator"] == "LeverageSHAP")
        # | (results_df["approximator"] == "PolySHAP-2ADD-10%")
        # | (results_df["approximator"] == "PolySHAP-2ADD-20%")
        | (results_df["approximator"] == "PolySHAP-2ADD-50%")
        # | (results_df["approximator"] == "PolySHAP-2ADD-75%")
        | (results_df["approximator"] == "PolySHAP-2ADD")
        # | (results_df["approximator"] == "PolySHAP-3ADD-20%")
        | (results_df["approximator"] == "PolySHAP-3ADD-50%")
        # | (results_df["approximator"] == "PolySHAP-3ADD-75%")
        | (results_df["approximator"] == "PolySHAP-3ADD")
        | (results_df["approximator"] == "PolySHAP-4ADD")
    ]

    plot_df = plot_df[plot_df["id_config_approximator"] == 39]

    for game_type in GAME_TYPES:
        plot_df_game_type = plot_df[results_df["game_type"] == game_type]
        for game_id in GAME_IDS:
            plot_df_game_id = plot_df_game_type[plot_df_game_type["game_id"] == game_id]
            if len(plot_df_game_id) > 0:
                metric = "MSE"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=True,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(f"plots/{game_type}/{game_id}_{metric}_standard.pdf")

                metric = "Precision@5"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=False,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(f"plots/{game_type}/{game_id}_{metric}_standard.pdf")

                metric = "Precision@10"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=False,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(f"plots/{game_type}/{game_id}_{metric}_standard.pdf")

                metric = "SpearmanCorrelation@10"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=False,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(f"plots/{game_type}/{game_id}_{metric}_standard.pdf")

                metric = "SpearmanCorrelation"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=False,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(f"plots/{game_type}/{game_id}_{metric}_standard.pdf")

    # Plot paired vs standard
    plot_df = results_df[
        # (results_df["approximator"] == "PermutationSampling")
        # (results_df["approximator"] == "RegressionMSR") |
        # | (results_df["approximator"] == "KernelSHAP")
        (results_df["approximator"] == "LeverageSHAP")
        # | (results_df["approximator"] == "PolySHAP-2ADD-10%")
        # | (results_df["approximator"] == "PolySHAP-2ADD-20%")
        # | (results_df["approximator"] == "PolySHAP-2ADD-50%")
        # | (results_df["approximator"] == "PolySHAP-2ADD-75%")
        | (results_df["approximator"] == "PolySHAP-2ADD")
        # | (results_df["approximator"] == "PolySHAP-3ADD-20%")
        # | (results_df["approximator"] == "PolySHAP-3ADD-50%")
        # | (results_df["approximator"] == "PolySHAP-3ADD-75%")
        | (results_df["approximator"] == "PolySHAP-3ADD")
        | (results_df["approximator"] == "PolySHAP-4ADD")
    ]

    config_id = [37, 39]  # [39, 37]

    if config_id is not None:
        plot_df = plot_df[plot_df["id_config_approximator"].isin(config_id)]

    for game_type in GAME_TYPES:
        plot_df_game_type = plot_df[plot_df["game_type"] == game_type]
        for game_id in GAME_IDS:
            plot_df_game_id = plot_df_game_type[plot_df_game_type["game_id"] == game_id]
            if len(plot_df_game_id) > 0:
                metric = "MSE"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=True,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(
                    f"plots/{game_type}/{game_id}_{metric}_paired_vs_standard.pdf"
                )

                metric = "Precision@5"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=False,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(
                    f"plots/{game_type}/{game_id}_{metric}_paired_vs_standard.pdf"
                )

                metric = "Precision@10"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=False,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(
                    f"plots/{game_type}/{game_id}_{metric}_paired_vs_standard.pdf"
                )

                metric = "SpearmanCorrelation@10"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=False,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(
                    f"plots/{game_type}/{game_id}_{metric}_paired_vs_standard.pdf"
                )

                metric = "SpearmanCorrelation"
                dataset = plot_df_game_id["game"].unique()[0]
                fig, ax = plot_approximation_quality(
                    data=plot_df_game_id,
                    metric=metric,
                    log_scale_y=False,
                    log_scale_x=False,
                    legend=False,
                )
                ax.set_title(DATA_NAMES[dataset], fontsize=TITLE_FONT_SIZE)
                fig.tight_layout()
                fig.savefig(
                    f"plots/{game_type}/{game_id}_{metric}_paired_vs_standard.pdf"
                )
