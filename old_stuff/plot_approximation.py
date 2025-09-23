"""This module plots the approximation quality of the different approximators."""

import os

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "../results"
os.makedirs(DATA_DIR, exist_ok=True)


if __name__ == "__main__":
    metric = "SpearmanCorrelation"

    # get data
    file_name = "results_vit_new.csv"
    data_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(data_path)

    # drop "image_name" column
    n_images = df["image_name"].nunique()
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
        plt.plot(x_val, y_val, label=approximator)
    plt.xlabel("Budget")
    plt.ylabel(metric)
    plt.legend()
    plt.title(file_name + f" (n: {n_images})")
    # plt.ylim(0, 0.03)

    plt.show()
