from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.special import binom

if __name__ == "__main__":
    # Load the results from the CSV file
    results_df = pd.read_csv("../experiments/results_benchmark.csv")
    results_df = results_df.sort_values(by="n_players")

    results_df = results_df[results_df["approximator"].str.contains("Lev1")]
    results_overfitting = results_df[results_df["game_type"] == "overfitting"]

    # Function to extract k from the string
    def extract_k(s):
        match = re.search(r"ShapleyGAX-(\d+)ADD", s)
        return int(match.group(1)) if match else None

    # Function to compute sum of binomial coefficients using binom
    def binom_sum(k, n):
        return int(sum(binom(n, i) for i in range(k + 1)))

    results_overfitting["approximator_order"] = results_overfitting[
        "approximator"
    ].apply(extract_k)
    results_overfitting["approximator_variables"] = results_overfitting.apply(
        lambda row: binom_sum(row["approximator_order"], row["n_players"]), axis=1
    )

    results_aggregated = (
        results_overfitting.groupby(
            [
                "game_id",
                "approximator",
                "n_players",
                "used_budget",
                "approximator_variables",
                "approximator_order",
            ]
        )["MSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    game_ids = results_aggregated["game_id"].unique()
    config_ids = results_aggregated["id_config_approximator"].unique()

    for game_id in game_ids:
        for config_id in config_ids:
            plot_df = results_aggregated[
                (results_aggregated["game_id"] == game_id)
                & (results_aggregated["id_config_approximator"] == config_id)
            ]
            # Compute standard error
            plot_df["SE"] = plot_df["std"] / np.sqrt(plot_df["count"])

            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            # Assuming your dataframe is called df
            # plt.figure(figsize=(10, 6), dpi=300)
            # plot_df = plot_df[plot_df["used_budget"] == 10000]

            plt.figure(figsize=(10, 6))
            budgets = plot_df["used_budget"].unique()
            colors = [
                "#1E3A8A",
                "#60A5FA",
                "#FCD34D",
                "#F87171",
            ]  # one color per category
            # Create x-axis labels: "binom_sum (k=...)"
            plot_df["x_label"] = plot_df.apply(
                lambda row: f"{row['approximator_variables']} (k={row['approximator_order']})",
                axis=1,
            )
            for budget, color in zip(budgets, colors, strict=False):
                subset = plot_df[plot_df["used_budget"] == budget]
                plt.errorbar(
                    subset["approximator_variables"],
                    subset["mean"],
                    yerr=subset["SE"],
                    fmt="o",
                    label=f"{budget}",
                    color=color,
                    capsize=5,
                )
                plt.axvline(
                    x=budget,
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                    label="_nolegend_",
                )
            plt.yscale(
                "log"
            )  # set y-axis to log scale    plt.xlabel("Sum of Binomials (binom_sum) [k in brackets]")
            plt.xscale("log")
            plt.ylabel("MSE")
            plt.xlabel("Size of Explanation Basis")
            plt.title(game_id, config_id)
            orders_for_budgets = plot_df.drop_duplicates(
                "approximator_variables"
            ).set_index("approximator_variables")["x_label"]
            plt.xticks(
                ticks=orders_for_budgets.index,  # positions = budgets
                labels=orders_for_budgets.values,  # labels = orders
                rotation=45,
                ha="right",
            )
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(title="Budget")
            plt.tight_layout()
            plt.show()
