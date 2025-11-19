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
    "CIFAR10": "CIFAR-10 ViT16 ($d=16$)"
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
    "PolySHAP-3ADD-dlog(d)": "3-PolySHAP (log)",
}

TITLE_FONT_SIZE = 24


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


def binomial_budget(n_players, k):
    return sum(math.comb(n_players, i) for i in range(1, k + 1))


def threshold_budget(n_players):
    if n_players <= 20:
        q = binomial_budget(n_players, 4)
        result = binomial_budget(n_players, 4) * 1
    if n_players > 20 and n_players < 40:
        q = binomial_budget(n_players, 3)
        result = binomial_budget(n_players, 3) * 1
    if n_players >= 40:
        q = binomial_budget(n_players, 2)
        result = binomial_budget(n_players, 2) * 1
    return int(result)


if __name__ == "__main__":
    """This script provides the preliminary performance table."""
    # Load the results from the CSV file
    results_df = pd.read_csv("results_benchmark.csv")
    results_df = results_df.sort_values(by="n_players")

    results_df[["p", "q"]] = results_df["approximator"].apply(
        lambda x: pd.Series(parse_approximator(x))
    )
    results_df["minimum_budget_to_plot"] = results_df.apply(compute_value, axis=1)
    results_df = results_df[
        results_df["used_budget"] >= results_df["minimum_budget_to_plot"]
    ]

    # select explanation games
    results_df = results_df[
        (results_df["approximator"] == "RegressionMSR")
        | (results_df["approximator"] == "PermutationSampling")
        | (results_df["approximator"] == "LeverageSHAP")
        | (results_df["approximator"] == "PolySHAP-2ADD-50%")
        | (results_df["approximator"] == "PolySHAP-2ADD")
        | (results_df["approximator"] == "PolySHAP-3ADD-50%")
        | (results_df["approximator"] == "PolySHAP-3ADD")
        | (results_df["approximator"] == "PolySHAP-4ADD")
        | (results_df["approximator"] == "PolySHAP-3ADD-dlog(d)")
        | (results_df["approximator"] == "MSR")
        | (results_df["approximator"] == "SVARM")
        ]

    results_df["min_table_budget"] = results_df["n_players"].apply(threshold_budget)

    results_df = results_df[results_df["used_budget"] >= results_df["min_table_budget"]]

    results_df = results_df.loc[
        results_df.groupby(
            ["game", "approximator", "id_config_approximator", "n_players"]
        )["used_budget"].transform("min")
        == results_df["used_budget"]
    ]

    results_df = (
        results_df.groupby(
            [
                "game",
                "approximator",
                "id_config_approximator",
                "used_budget",
                "n_players",
            ]
        )
        .agg(  # 1st 2nd and 3rth quartile and mean)
            MSE_mean=("MSE", "mean"),
            MSE_q1=("MSE", lambda x: np.percentile(x, 25)),
            MSE_q2=("MSE", lambda x: np.percentile(x, 50)),
            MSE_q3=("MSE", lambda x: np.percentile(x, 75)),
        )
        .reset_index()
    )

    results_df["game_id"] = results_df["game"].map(DATA_NAMES)
    results_df = results_df.drop(columns=["game"])
    # results_df["approximator"] = results_df["approximator"].map(APPROXIMATOR_RENAMING)

    results_df.to_csv("performance_table.csv", index=False)
