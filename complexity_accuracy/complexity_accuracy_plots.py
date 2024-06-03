import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_interaction_sizes_errors(results, weighting_scheme):
    plt.figure()
    for interaction_size in INTERACTION_RANGE:
        game_id = "SOUM_" + str(interaction_size)
        for index in INDICES:
            plt.subplot(3, 3, interaction_size - 1)
            plt.plot(
                results.loc[game_id, weighting_scheme, index].index,
                results.loc[game_id, weighting_scheme, index]["l2"],
                label=index,
                color=INDEX_COLORS[index],
            )
        plt.title("Size " + str(interaction_size))
        plt.xticks(range(1, 11))
    # plt.legend()
    plt.tight_layout()
    plt.savefig("plots/l2_" + weighting_scheme + "_single_interaction_by_size.png")
    plt.show()


def plot_interaction_sizes_r2(results, weighting_scheme):
    plt.figure()
    for interaction_size in INTERACTION_RANGE:
        game_id = "SOUM_" + str(interaction_size)
        for index in INDICES:
            plt.subplot(3, 3, interaction_size - 1)
            plt.plot(
                results.loc["SOUM", game_id, weighting_scheme, index].index,
                results.loc["SOUM", game_id, weighting_scheme, index]["r2"],
                label=index,
                color=INDEX_COLORS[index],
            )
        if interaction_size > 7:
            plt.text(
                2,
                0.93,
                "SHAP:\n "
                + str(
                    np.round(
                        results.loc["SOUM", game_id, weighting_scheme, index, 1]["r2"].astype(
                            float
                        ),
                        2,
                    )
                ),
                fontsize=8,
                color="black",
                ha="center",
            )
        else:
            plt.text(
                9,
                0.65,
                "SHAP:\n "
                + str(
                    np.round(
                        results.loc["SOUM", game_id, weighting_scheme, index, 1]["r2"].astype(
                            float
                        ),
                        2,
                    )
                ),
                fontsize=8,
                color="black",
                ha="center",
            )
        plt.title("Size " + str(interaction_size))
        plt.xticks(range(1, 11))
        plt.ylim(0.6, 1.05)
    # plt.legend()
    plt.tight_layout()
    plt.savefig("plots/r2_" + weighting_scheme + "_single_interaction_by_size.png")
    plt.show()


def plot_errors(errors_mean, errors_std, desc):
    plt.figure()
    for index in INDICES:
        plt.plot(
            errors_mean.loc[index].index,
            errors_mean.loc[index],
            label=index,
            color=INDEX_COLORS[index],
        )
        plt.fill_between(
            errors_mean.loc[index].index,
            errors_mean.loc[index] - errors_std.loc[index],
            errors_mean.loc[index] + errors_std.loc[index],
            color=INDEX_COLORS[index],
            alpha=0.2,
            label="",
        )

    plt.legend()
    plt.title(desc)
    plt.xlabel(XLABEL)
    plt.ylabel("Weighted Squared Loss (" + weighting_scheme + ")")
    plt.savefig("plots/l2_" + weighting_scheme + "_" + desc + ".png")
    plt.show()


def plot_r2(r2_means, r2_stds, desc):
    plt.figure()
    for index in INDICES:
        plt.plot(
            r2_means[index].index,
            r2_means[index],
            label=index,
            color=INDEX_COLORS[index],
        )
        plt.fill_between(
            r2_means.loc[index].index,
            r2_means.loc[index] - r2_stds.loc[index],
            r2_means.loc[index] + r2_stds.loc[index],
            color=INDEX_COLORS[index],
            alpha=0.2,
            label="",
        )
        plt.xticks(range(1, 11))
        plt.ylim(0, 1.05)
        plt.xticks(range(np.min(r2_means.loc[index].index), np.max(r2_means.loc[index].index) + 1))
    plt.legend()
    plt.title(desc)
    plt.xlabel("Explanation Order")
    plt.ylabel("Weighted R2 (" + weighting_scheme + ")")
    plt.savefig("plots/r2_" + weighting_scheme + "_" + desc + ".png")
    plt.show()


def plot_r2_3by3(results):
    plot_pos = 1
    for weighting_scheme in WEIGHTING_SCHEMES:
        plt.figure()
        for game_title, game_title_short in PLOT_LIST.items():
            plt.subplot(3, 3, plot_pos)
            current_results = results.loc[game_title]
            means = current_results.groupby(["weighting_scheme", "index", "order"])[["r2"]].mean()
            sems = current_results.groupby(["weighting_scheme", "index", "order"])[["r2"]].sem()
            for index in INDICES:
                plt.plot(
                    means.loc[weighting_scheme, index].index,
                    means.loc[weighting_scheme, index],
                    label=index,
                    color=INDEX_COLORS[index],
                )
                plt.fill_between(
                    means.loc[weighting_scheme, index].index,
                    means.loc[weighting_scheme, index]["r2"]
                    - sems.loc[weighting_scheme, index]["r2"],
                    means.loc[weighting_scheme, index]["r2"]
                    + sems.loc[weighting_scheme, index]["r2"],
                    color=INDEX_COLORS[index],
                    alpha=0.2,
                    label="",
                )
            plt.ylim(0, 1.05)
            plt.xticks(
                range(
                    np.min(means.loc[weighting_scheme, index].index),
                    np.max(means.loc[weighting_scheme, index].index) + 1,
                )
            )
            plt.title(game_title_short)
            plot_pos += 1

        plt.tight_layout()
        plt.savefig("plots/r2_" + weighting_scheme + ".png")
        plt.show()


def load_results(errors, weighted_r2, identifier):
    tmp = []
    for weighting_scheme in WEIGHTING_SCHEMES:
        for index in INDICES:
            df = pd.DataFrame()
            df["l2"] = errors[weighting_scheme][index]
            df["r2"] = weighted_r2[weighting_scheme][index]
            df["order"] = df.index
            df["index"] = index
            df["weighting_scheme"] = weighting_scheme
            tmp.append(df)
        df_results = pd.concat(tmp)
        df_results.to_csv("results/" + identifier + ".csv")


if __name__ == "__main__":
    INDICES = ["k-SII", "STII", "FSII", "FBII"]
    INDEX_COLORS = {
        "FBII": "#ef27a6",
        "FSII": "#7d53de",
        "STII": "#00b4d8",
        "k-SII": "#ff6f00",
        "SII": "#ffba08",
    }
    XLABEL = "Explanation Order"

    WEIGHTING_SCHEMES = ["Shapley kernel"]  # ["uniform", "Shapley kernel"]
    PLOT_L2 = False
    PLOT_R2 = True

    PLOT_SYNTHETIC_INTERACTION_EXPERIMENT = True
    PLOT_BENCHMARK_GAMES_EXPERIMENT = False

    directory_path = "results/"

    dataframes = []
    game_list = []
    # Loop over the list of files in the directory
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path) and file.endswith(".csv"):
            # Read the CSV file into a DataFrame
            game_id = file.split("/")[-1][:-4]
            df = pd.read_csv(file_path)
            df["game_id"] = game_id
            dataframes.append(df)
            game_list.append(game_id)

    results = pd.concat(dataframes)
    results = results.set_index(["game_title", "game_id", "weighting_scheme", "index", "order"])

    if PLOT_SYNTHETIC_INTERACTION_EXPERIMENT:
        # Plot single interaction plots
        INTERACTION_RANGE = range(2, 11)
        for weighting_scheme in WEIGHTING_SCHEMES:
            if PLOT_L2:
                plot_interaction_sizes_errors(results, weighting_scheme)
            if PLOT_R2:
                plot_interaction_sizes_r2(results, weighting_scheme)

    if PLOT_BENCHMARK_GAMES_EXPERIMENT:
        PLOT_LIST = {
            "California Housing Global XAI": "CH Global XAI",
            "California Housing Local XAI": "CH Local XAI",
            "Image Classifier Local XAI": "Image Local XAI",
            "Bike Sharing Ensemble Selection": "Bike Ensemble Sel.",
            "Bike Sharing Dataset Valuation": "Bike Dataset Val.",
        }
        # Plot average over classes
        for game_title in results.index.levels[0]:
            current_results = results.loc[game_title]
            current_results_means = current_results.groupby(["weighting_scheme", "index", "order"])[
                ["l2", "r2"]
            ].mean()
            current_results_sems = current_results.groupby(["weighting_scheme", "index", "order"])[
                ["l2", "r2"]
            ].sem()
            for weighting_scheme in WEIGHTING_SCHEMES:
                if PLOT_L2:
                    plot_errors(
                        current_results_means.loc[weighting_scheme]["l2"],
                        current_results_sems.loc[weighting_scheme]["l2"],
                        game_title,
                    )
                if PLOT_R2:
                    plot_r2(
                        current_results_means.loc[weighting_scheme]["r2"],
                        current_results_sems.loc[weighting_scheme]["r2"],
                        game_title,
                    )

        plot_r2_3by3(
            results,
        )
