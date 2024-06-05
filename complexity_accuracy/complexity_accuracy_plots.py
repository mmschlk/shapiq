import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import viridis


def plot_interaction_sizes_errors(results, weighting_scheme):
    plt.figure()
    for pos, interaction_size in enumerate(INTERACTION_RANGE):
        game_id = "SOUM_" + str(interaction_size)
        for index in INDICES:
            plt.subplot(2, 3, pos + 1)
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
    plot_pos = 0
    for _, interaction_size in enumerate(INTERACTION_RANGE):
        if interaction_size not in (4, 5, 6, 8):
            continue
        game_id = "SOUM_" + str(interaction_size)
        for index in INDICES:
            plt.subplot(1, 4, plot_pos + 1)
            plt.plot(
                results.loc["SOUM", game_id, weighting_scheme, index].index,
                results.loc["SOUM", game_id, weighting_scheme, index]["r2"],
                label=index,
                color=INDEX_COLORS[index],
                marker="o",
                markersize=3,
            )
        plt.text(
            10.5,
            0.35,
            "SHAP: "
            + str(
                np.round(
                    results.loc["SOUM", game_id, weighting_scheme, "FSII", 1]["r2"].astype(float),
                    2,
                )
            ),
            fontsize=11,
            color="black",
            ha="right",
        )

        plt.title("Size " + str(interaction_size))
        plt.xticks(range(1, 11))
        plt.xlim(0, 11)
        plt.ylim(0.3, 1.05)
        if plot_pos + 1 == 1:
            plt.gca().set_yticks(np.linspace(0.4, 1, 4))
            plt.ylabel("Weighted R2", fontsize=12)
        else:
            plt.gca().set_yticks([])
        plot_pos += 1
        # plt.legend()
        # Adjust the space between subplots

        # add x-label to the whole plot as text saying "Explanation Order"
    plt.text(
        -11,
        0.13,
        "Explanation Order",
        fontsize=12,
        color="black",
        ha="center",
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("plots_new/r2_" + weighting_scheme + "_single_interaction_by_size.pdf")
    plt.show()


def plot_r2_1by4(results):
    """Does the same thing as plot_r2_2by3 but with 1 row and 4 columns"""
    plot_pos = 1
    for weighting_scheme in WEIGHTING_SCHEMES:
        plt.figure()
        for game_title, game_title_short in PLOT_LIST.items():
            plt.subplot(1, 4, plot_pos)
            current_results = results.loc[game_title]
            means = current_results.groupby(["weighting_scheme", "index", "order"])[["r2"]].mean()
            sems = current_results.groupby(["weighting_scheme", "index", "order"])[["r2"]].sem()
            for index in INDICES:
                plt.plot(
                    means.loc[weighting_scheme, index].index,
                    means.loc[weighting_scheme, index],
                    label=index,
                    color=INDEX_COLORS[index],
                    marker="o",
                    markersize=3,
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
            plt.ylim(0.3, 1.05)
            if plot_pos == 1:
                plt.gca().set_yticks(np.linspace(0.4, 1, 4))
            else:
                plt.gca().set_yticks([])

            plt.xticks(
                range(
                    1,
                    # np.min(means.loc[weighting_scheme, index].index),
                    # np.max(means.loc[weighting_scheme, index].index) + 1 ,
                    np.max(means.loc[weighting_scheme, index].index) + 1,
                )
            )
            x_text = 0.5 if plot_pos > 2 else np.max(means.loc[weighting_scheme, index].index)
            y_text = 0.92 if plot_pos > 2 else 0.35
            align = "left" if plot_pos > 2 else "right"
            sep_str = "\n" if plot_pos > 2 else " "
            plt.text(
                x_text,
                y_text,
                "SHAP:"
                + sep_str
                + str(
                    np.round(
                        means.loc[weighting_scheme, "FSII", 1]["r2"].astype(float),
                        2,
                    )
                ),
                fontsize=11,
                color="black",
                ha=align,
            )

            # get xticks
            # get curre_top_lim
            if plot_pos == 1:
                plt.ylabel("Weighted R2", fontsize=12)

            top_lim = plt.gca().get_xlim()[1]
            plt.xlim(0, top_lim * 1.05)

            plt.title(game_title_short)
            plot_pos += 1

        # add x-label to the whole plot as text saying "Explanation Order"
        plt.text(
            -11,
            0.13,
            "Explanation Order",
            fontsize=12,
            color="black",
            ha="center",
        )

        # Adjust the space between subplots
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("plots_new/r2_" + weighting_scheme + ".pdf")
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

    # adjust figure size
    plt.rcParams["figure.figsize"] = (7.4, 3)

    INDICES = ["k-SII", "STII", "FSII", "FBII"]
    colors = viridis(np.linspace(0, 1, 4))
    INDEX_COLORS = {
        "FBII": "#ef27a6",  # "#ef27a6",
        "FSII": "#7d53de",  # "#7d53de",
        "STII": "#00b4d8",  # "#00b4d8",
        "k-SII": "#ff6f00",  # "#ff6f00",
        "SII": "#ffba08",
    }
    XLABEL = "Explanation Order"

    WEIGHTING_SCHEMES = ["Shapley kernel"]  # ["uniform", "Shapley kernel"]
    PLOT_L2 = False
    PLOT_R2 = True

    PLOT_SYNTHETIC_INTERACTION_EXPERIMENT = True
    PLOT_BENCHMARK_GAMES_EXPERIMENT = True

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
        INTERACTION_RANGE = range(4, 10)
        for weighting_scheme in WEIGHTING_SCHEMES:
            if PLOT_L2:
                plot_interaction_sizes_errors(results, weighting_scheme)
            if PLOT_R2:
                plot_interaction_sizes_r2(results, weighting_scheme)

    if PLOT_BENCHMARK_GAMES_EXPERIMENT:
        PLOT_LIST = {
            "Image Classifier Local XAI": "Image Local XAI",
            "California Housing Local XAI": "CH Local XAI",
            "California Housing Global XAI": "CH Global XAI",
            "California Housing Dataset Valuation": "CH Dataset Val.",
            # "Adult Census Ensemble Selection": "Adult Ensemble Sel.",
            # "Adult Census Dataset Valuation": "Adult Dataset Val.",
            # "California Housing Ensemble Selection": "CH Ensemble Sel.",
            # "Bike Sharing Local XAI": "Bike LocalXAI",
            # "Bike Sharing Ensemble Selection": "Bike Ensemble Sel.",
            # "Bike Sharing Dataset Valuation": "Bike Dataset Val.",
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
                pass
                # if PLOT_L2:
                #     plot_errors(
                #         current_results_means.loc[weighting_scheme]["l2"],
                #         current_results_sems.loc[weighting_scheme]["l2"],
                #         game_title,
                #     )
                # if PLOT_R2:
                #     plot_r2(
                #         current_results_means.loc[weighting_scheme]["r2"],
                #         current_results_sems.loc[weighting_scheme]["r2"],
                #         game_title,
                #     )

        plot_r2_1by4(
            results,
        )
