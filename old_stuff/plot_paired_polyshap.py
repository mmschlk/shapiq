from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.special import comb
from shapiq.benchmark import plot_approximation_quality
from shapiq.benchmark.plot import plot_pairing_vs_standard
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the results from the CSV file
    results_df = pd.read_csv("../experiments/results_benchmark.csv")
    results_df = results_df.sort_values(by="n_players")

    results_df = results_df[
        (results_df["approximator"] == "PermutationSampling")
        # | (results_df["approximator"] == "KernelSHAP")
        | (results_df["approximator"] == "LeverageSHAP")
        # | (results_df["approximator"] == "PolySHAP-2ADD-10%")
        # | (results_df["approximator"] == "PolySHAP-2ADD-20%")
        # | (results_df["approximator"] == "PolySHAP-2ADD-50%")
        # | (results_df["approximator"] == "PolySHAP-2ADD-75%")
        | (results_df["approximator"] == "PolySHAP-2ADD")
        # | (results_df["approximator"] == "PolySHAP-3ADD")
        # | (results_df["approximator"] == "PolySHAP-4ADD")
    ]

    # results_df = results_df[results_df["model"] == "random_forest"]

    GAME_IDS = results_df["game_id"].unique()
    GAME_TYPES = results_df["game_type"].unique()
    metric = "MSE"

    config_id = None
    config_id = [39, 37]

    if config_id is not None:
        results_df = results_df[results_df["id_config_approximator"].isin(config_id)]

    data_order = (
        results_df.groupby(
            [
                "game_id",
                "approximator",
                "used_budget",
                "iteration",
                "id_config_approximator",
                "n_players",
            ]
        )
        .agg(
            {
                metric: [
                    "mean",
                    "std",
                    "var",
                    "count",
                    "median",
                ],
            },
        )
        .reset_index()
    )
    # rename the columns of grouped data
    new_columns = [
        "_".join(col).strip() if col[1] != "" else col[0] for col in data_order.columns
    ]
    new_columns = [col.replace(f"{metric}_", "") for col in new_columns]

    data_order.columns = new_columns

    data_order["budget_threshold"] = (
        (comb(data_order["n_players"], 2) + data_order["n_players"])
        * np.log((comb(data_order["n_players"], 2) + data_order["n_players"]))
    ).astype(int)

    # Filter: keep only rows where budget >= threshold
    data_order = data_order[data_order["used_budget"] >= data_order["budget_threshold"]]
    data_plot = data_order.loc[
        data_order.groupby(
            [
                "game_id",
                "approximator",
                "iteration",
                "id_config_approximator",
                "n_players",
            ]
        )["used_budget"].idxmin()
    ]

    data_plot["sem"] = (
        data_plot["std"] / data_plot["count"] ** 0.5
    )  # compute standard error

    # Extract relevant columns
    plot_df = data_plot[
        [
            "game_id",
            "id_config_approximator",
            "approximator",
            "mean",
            "sem",
        ]
    ].copy()

    # Set a MultiIndex (game_id, id_config_approximator)
    plot_df.set_index(
        ["game_id", "id_config_approximator", "approximator"], inplace=True
    )
    plot_pairing_vs_standard(plot_df)
