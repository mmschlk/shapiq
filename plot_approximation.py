"""This module plots the approximation quality of the different approximators."""

import os

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)


if __name__ == "__main__":

    metric = "SpearmanCorrelation@50"

    # get data
    data_path = os.path.join(DATA_DIR, "results_vit.csv")
    df = pd.read_csv(data_path)

    # drop "image_name" column
    df = df.drop(columns=["image_name"])
    all_columns = list(df.columns)
    all_metrics = [col for col in all_columns if col not in ["approximator", "budget"]]
    print("All available metrics: ", all_metrics)
    if metric not in all_metrics:
        raise ValueError(f"Metric {metric} not in columns {all_metrics}")

    # aggregate the data for all "approximator" and "budget" combinations
    df_agg = df.groupby(["approximator", "budget"]).mean().reset_index()

    # plot the metric for all approximators (different color) for all budgets
    plt.figure(figsize=(10, 6))
    for approximator in df_agg["approximator"].unique():
        df_approx = df_agg[df_agg["approximator"] == approximator]
        x_val = df_approx["budget"].values
        y_val = df_approx[metric].values
        plt.scatter(x_val, y_val, label=approximator)
    plt.xlabel("Budget")
    plt.ylabel(metric)
    plt.legend()
    plt.show()
