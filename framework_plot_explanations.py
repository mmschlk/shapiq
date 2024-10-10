"""This script plots the explanations for a selection of games."""

import os
from itertools import product

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from framework_utils import get_save_name

RESULTS_DIR = "framework_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


COLOR_PALETTE = ["#00b4d8", "#ef27a6", "#ff6f00", "#ffbe0b"]


def load_explanation_data(
    num_samples: int = 10_000,
    model_name: str = "lin_reg",
    rho_values: list[float] = (0.0, 0.5, 0.9),
    sample_sizes: list[int] = (512,),
    interaction_data: bool = False,
    ones_instances: bool = (True,),
    n_instances: list[int] = (1,),
    random_seeds: int = 30,
    only_load: bool = False,
) -> pd.DataFrame:
    """Loads the explanation data from disk."""
    int_str = "int" if interaction_data else "no_int"

    if only_load:
        try:
            data_all_df = pd.read_csv(os.path.join(RESULTS_DIR, f"all_explanations_{int_str}.csv"))
            return data_all_df
        except FileNotFoundError:
            pass

    random_seeds = list(range(random_seeds))

    # params games
    rho_values = list(rho_values)
    n_instances = list(n_instances)
    sample_sizes = list(sample_sizes)
    ones_instances = [ones_instances]

    data_settings = list(
        product(random_seeds, rho_values, n_instances, sample_sizes, ones_instances)
    )

    data_all: list[pd.DataFrame] = []
    for random_seed, rho_value, n_instance, sample_size, ones_instance in tqdm(data_settings):
        # get the save name
        save_name = get_save_name(
            interaction_data=interaction_data,
            model_name=model_name,
            random_seed=random_seed,
            num_samples=num_samples,
            rho=rho_value,
            fanova="all",
            sample_size=sample_size,
            instance_id=0,
            data_name="synthetic_ones" if ones_instance else "synthetic",
        )
        save_path = os.path.join(RESULTS_DIR, f"{save_name}_explanations.csv")

        # load the explanations
        explanations_df = pd.read_csv(save_path)
        explanations_df["random_seed"] = random_seed
        explanations_df["rho"] = rho_value
        explanations_df["n_instance"] = n_instance
        explanations_df["sample_size"] = sample_size
        data_all.append(explanations_df)

    # save all data
    data_all_df = pd.concat(data_all)
    data_all_df.to_csv(os.path.join(RESULTS_DIR, f"all_explanations_{int_str}.csv"), index=False)
    return data_all_df


def plot_bar_plot(
    df: pd.DataFrame,
    axis: plt.Axes,
    feature_sets: list[tuple[int, ...]],
    group_ordering: list[dict[str, list]],
    spacing: float = 0.1,
    bar_padding: float = 0.05,
    inner_group_spacing: float = 0.15,
    bar_width: float = 0.1,
    color_features: bool = True,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Plots the bar plot for the selected explanations."""
    group_ordering = list(group_ordering)
    rho_values = [element["rho_values"] for element in group_ordering if "rho_values" in element][0]
    fanova_settings = [
        element["fanova_setting"] for element in group_ordering if "fanova_setting" in element
    ][0]
    entities = [element["entity"] for element in group_ordering if "entity" in element][0]
    feature_influences = [
        element["feature_influence"] for element in group_ordering if "feature_influence" in element
    ][0]

    df_plot = df.copy()
    feature_sets = [str(feature_set) for feature_set in feature_sets]
    df_plot = df_plot[df_plot["feature_set"].isin(feature_sets)]
    df_plot = df_plot[df_plot["fanova_setting"].isin(fanova_settings)]
    df_plot = df_plot[df_plot["entity"].isin(entities)]
    df_plot = df_plot[df_plot["feature_influence"].isin(feature_influences)]
    df_plot = df_plot[df_plot["rho"].isin(rho_values)]

    grouping_cols = ["feature_set", "fanova_setting", "entity", "feature_influence", "rho"]

    # get mean and std of explanations
    df_agg = df_plot.groupby(grouping_cols)["explanation"].agg(["mean", "std"]).reset_index()

    # error if there are duplicates
    if len(df_plot) != len(df_plot.drop_duplicates()):
        raise ValueError("There are duplicates in the data.")

    # order the groups of bars with the group ordering
    n_inner_most_values = len(rho_values)
    plotting_order = []
    for entity in entities:
        for feature_influence in feature_influences:
            for fanova_setting in fanova_settings:
                for rho_value in rho_values:
                    group = {
                        "entity": entity,
                        "feature_influence": feature_influence,
                        "fanova_setting": fanova_setting,
                        "rho": rho_value,
                    }
                    plotting_order.append(group)
    print("Order", plotting_order)

    # ad horizontal lines at 0, 1, and 2
    for line in [0, 1, 2]:
        axis.axhline(line, color="gray", linestyle="--", alpha=0.75, linewidth=1, zorder=1)

    min_height, max_height = 0, 0
    all_positions = []
    group_pos_endings = []
    x_pos = 0
    x_pos_line = x_pos - inner_group_spacing / 2 - spacing / 2 - bar_padding / 2 - bar_width / 2
    group_start = [x_pos_line]
    group_end = []
    plt.vlines(x=x_pos_line, ymin=0, ymax=3, color="black", linestyle="--")
    color_id = 0
    for group_id, group in enumerate(plotting_order, start=1):

        color = COLOR_PALETTE[color_id]
        if group_id % n_inner_most_values == 0:
            color_id += 1

        fanova_setting = group["fanova_setting"]
        entity = group["entity"]
        feature_influence = group["feature_influence"]
        rho_value = group["rho"]

        df_group = df_agg[
            (df_agg["fanova_setting"] == fanova_setting)
            & (df_agg["entity"] == entity)
            & (df_agg["feature_influence"] == feature_influence)
            & (df_agg["rho"] == rho_value)
        ]

        # get the x values
        for f_id, feature_set in enumerate(feature_sets):
            y_value = float(df_group[df_group["feature_set"] == feature_set]["mean"].values[0])
            y_error = float(df_group[df_group["feature_set"] == feature_set]["std"].values[0])
            max_height = max(max_height, y_value + y_error)
            min_height = min(min_height, y_value - y_error)
            if color_features:
                color = COLOR_PALETTE[f_id]
            # add to plot
            axis.bar(x=x_pos, height=y_value, width=bar_width, yerr=y_error, color=color)
            all_positions.append(x_pos)
            x_pos += bar_width
            x_pos += bar_padding
        x_pos += spacing

        if group_id % n_inner_most_values == 0:
            x_pos += inner_group_spacing

            x_pos_line = (
                x_pos - inner_group_spacing / 2 - spacing / 2 - bar_padding / 2 - bar_width / 2
            )
            group_end.append(x_pos_line)
            group_start.append(x_pos_line)

    group_start = group_start[:-1]

    # add accent behind groups
    group_pos_endings.append(x_pos)
    for i, (start, end) in enumerate(zip(group_start, group_end)):
        if i % 2 == 0:
            continue
        axis.add_patch(
            plt.Rectangle(
                (start, -50),
                end - start,
                100,
                color="black",
                alpha=0.75,
                zorder=0,
            )
        )

    # set the ylim
    y_lim = (min_height - 0.1, max_height + 0.1)
    x_lim = (group_start[0], group_end[-1])

    axis.set_xticks([])
    axis.set_yticks([])

    return y_lim, x_lim


if __name__ == "__main__":

    # load the data
    _ = load_explanation_data(only_load=True, interaction_data=False)
    data = load_explanation_data(only_load=True, interaction_data=True)

    FANOVA_SETTINGS = ["c", "m", "b"]
    FEATURE_INFLUENCES = ["pure", "partial", "full"]
    RHO_VALUES = [0.0, 0.5, 0.9]

    pad = True
    figsize = (12, 10)
    title_fontsize = 25
    label_fontsize = 17

    # plot the explanations
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=False, sharey=False)
    y_lim_min, y_lim_max = 0, 0
    x_lim_min, x_lim_max = 0, 0
    for i, fanova_setting in enumerate(FANOVA_SETTINGS):
        for j, feature_influence in enumerate(FEATURE_INFLUENCES):
            y_lim, x_lim = plot_bar_plot(
                df=data,
                axis=axes[i, j],
                feature_sets=[(0,), (1,), (2,), (3,)],
                group_ordering=[
                    {"entity": ["individual"]},
                    {"feature_influence": [feature_influence]},
                    {"fanova_setting": [fanova_setting]},
                    {"rho_values": RHO_VALUES},
                ],
                bar_width=0.15,
                spacing=0.1,
                inner_group_spacing=0.15,
                bar_padding=0.05,
            )
            y_lim_min = min(y_lim_min, y_lim[0])
            y_lim_max = max(y_lim_max, y_lim[1])
            x_lim_min = min(x_lim_min, x_lim[0])
            x_lim_max = max(x_lim_max, x_lim[1])

    # adjust the limits
    for i in range(3):
        for j in range(3):
            axes[i, j].set_ylim(y_lim_min, y_lim_max)
            axes[i, j].set_xlim(x_lim_min, x_lim_max)

    stops = np.linspace(x_lim_min, x_lim_max - 0.1, 4)[[1, 2, 3]]
    for i in range(3):
        for j in range(3):
            # add grey patch betweet stop 1 and 2
            axes[i, j].add_patch(
                plt.Rectangle(
                    (stops[0], -50),
                    stops[1] - stops[0],
                    100,
                    color="#f0f0f0",
                    alpha=0.5,
                    zorder=0,
                )
            )
            # add grey patch betweet stop 2 and 3
            axes[i, j].add_patch(
                plt.Rectangle(
                    (stops[1], -50),
                    stops[2] - stops[1],
                    100,
                    color="#f0f0f0",
                    alpha=1,
                    zorder=0,
                )
            )

    # add x-ticks to the top row and to the top of the plots
    x_tick_labels = RHO_VALUES
    stops = np.linspace(x_lim_min, x_lim_max, 7)[[1, 3, 5]]
    for i in range(3):
        for j in range(3):
            axes[i, j].yaxis.set_ticks_position("right")
            axes[i, j].xaxis.set_ticks_position("top")
            # remove current x-ticks
            axes[i, j].set_xticks(stops)
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticks(list(range(7)))
            axes[i, j].set_yticklabels([])

    for i in range(3):
        axes[0, i].set_xticklabels(x_tick_labels, fontsize=label_fontsize)

    # label position top
    axes[0, 1].xaxis.set_label_position("top")
    axes[0, 1].set_xlabel("Correlation", fontsize=title_fontsize, labelpad=10)

    for i in range(3):
        axes[i, 2].set_yticklabels(list(range(7)), fontsize=label_fontsize)

    # label position right
    axes[1, 2].yaxis.set_label_position("right")
    axes[1, 2].set_ylabel(
        "Explanation", fontsize=title_fontsize, labelpad=20, rotation=270, va="center"
    )

    # add the labels to the bottom row (feature influence)
    for i, feature_influence in enumerate(FEATURE_INFLUENCES):
        # capitalize the first letter
        feature_influence = feature_influence.capitalize()
        axes[2, i].set_xlabel(feature_influence, fontsize=title_fontsize, labelpad=10)

    # add the labels to the left column (fanova setting)
    for i, fanova_setting in enumerate(FANOVA_SETTINGS):
        axes[i, 0].set_ylabel(fanova_setting, fontsize=title_fontsize, labelpad=12, ha="right")

    # add super x label
    axes[2, 1].set_xlabel(
        "Partial\n$\\bf{Higher-order\ Interaction\ Influence}$",
        fontsize=title_fontsize,
        labelpad=10,
    )
    axes[1, 0].set_ylabel(
        "$\\bf{Influnece\ of\ Feature\ Distribution}$\nm",
        fontsize=title_fontsize,
        labelpad=12,
        ha="center",
    )

    # remove whitespace between subplots
    plt.tight_layout()
    if pad:
        pad_str = "pad"
        plt.subplots_adjust(wspace=0.08, hspace=0.08)
    else:
        pad_str = "no_pad"
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"explanations_all_{pad_str}_{str(figsize[0])}x{str(figsize[1])}.pdf")
    plt.show()

    # plot_bar_plot(
    #     df=data,
    #     fanova_settings=["b", "m", "c"],
    #     entities=["individual"],
    #     feature_sets=[(0,), (1,), (2,), (3,)],
    #     feature_influences=["pure", "partial", "full"],
    #     rho_values=[0.0],
    #     facet=True,
    # )
