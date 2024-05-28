"""This script loads the results json files and summarizes them in a pandas DataFrame."""

import os

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    name = "SentimentAnalysis_Game_mask_strategy=mask_k-SII_2.json"
    path = os.path.join("results", name)
    df = pd.read_json(path)
    print(df)

    approx_df = df[df["approximator"] != "Exact"]
    approx_df = approx_df.groupby(["approximator", "budget"]).agg({"MSE": "mean"}).reset_index()
    print(approx_df)

    # plot mse over budget for each approximator
    fig, ax = plt.subplots()
    for approximator in approx_df["approximator"].unique():
        plot_df = approx_df[approx_df["approximator"] == approximator]
        ax.plot(plot_df["budget"], plot_df["MSE"], label=approximator)
    # log scale
    ax.set_yscale("log")
    ax.set_xlabel("Budget")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.show()
