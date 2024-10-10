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
    fanova_settings: list[str],
    entities: list[str],
    feature_sets: list[tuple[int, ...]],
    feature_influences: list[str],
    rho_values: list[float],
    spacing: float = 0.5,
    bar_padding: float = 0.2,
) -> None:
    """Plots the bar plot for the selected explanations."""
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

    fig, axis = plt.subplots(figsize=(11, 4))

    # get bar width
    bar_space = 1 - spacing
    bar_width = bar_space / len(feature_sets)
    bar_padding = bar_width * bar_padding
    bar_width = bar_width - bar_padding

    # will be plotted in this order
    groups_of_bars = list(product(entities, feature_influences, fanova_settings, rho_values))
    print(groups_of_bars)
    n_groups = len(groups_of_bars)

    # get the color palette
    color_palette = plt.cm.viridis(np.linspace(0, 1, int(len(groups_of_bars) / 2)))

    min_height, max_height = 0, 0
    legend_items = {}

    for group_id, group in enumerate(groups_of_bars):
        entity, feature_influence, fanova_setting, rho_value = group

        color = color_palette[group_id // 2]
        x_pos = group_id + spacing / 2

        legend_name = f"{fanova_setting} {entity} {feature_influence} {rho_value}"
        legend_items[legend_name] = color

        df_group = df_agg[
            (df_agg["fanova_setting"] == fanova_setting)
            & (df_agg["entity"] == entity)
            & (df_agg["feature_influence"] == feature_influence)
            & (df_agg["rho"] == rho_value)
        ]

        # get the x values
        for feature_set in feature_sets:
            x_pos += bar_padding / 2 + bar_width / 2
            y_value = float(df_group[df_group["feature_set"] == feature_set]["mean"].values[0])
            y_error = float(df_group[df_group["feature_set"] == feature_set]["std"].values[0])
            max_height = max(max_height, y_value + y_error)
            min_height = min(min_height, y_value - y_error)

            # plot the bar
            axis.bar(x=x_pos, height=y_value, width=bar_width, yerr=y_error, color=color)
            x_pos += bar_padding / 2 + bar_width / 2 + bar_padding / 2

    # add a grey rectangles for each interaction order
    for i in range(n_groups):
        if i % 2 == 0:
            continue
        axis.add_patch(plt.Rectangle((i, -50), 1, 100, color="#eeeeee", alpha=0.5, zorder=0))

    for legend_item in legend_items:
        axis.plot([], [], color=legend_items[legend_item], label=legend_item)

    # set the ylim
    axis.set_ylim(min_height - 0.1, max_height + 2)

    # set xlim
    axis.set_xlim(0, n_groups)

    # ad horizontal lines at 0, 1, and 2
    for line in [0, 1, 2]:
        axis.axhline(line, color="#eeeeee", linestyle="--", alpha=0.75)

    # remove all xticks
    axis.set_xticks(np.arange(n_groups) + 0.5)
    axis.set_xticklabels([])

    # plot
    axis.set_ylabel("Explanation")
    axis.legend(ncols=3)
    plt.show()


if __name__ == "__main__":

    # load the data
    _ = load_explanation_data(only_load=False, interaction_data=False)
    data = load_explanation_data(only_load=False, interaction_data=True)

    # plot the explanations
    plot_bar_plot(
        df=data,
        fanova_settings=["c", "m", "b"],
        entities=["individual"],
        feature_sets=[(0,), (1,), (2,), (3,)],
        feature_influences=["partial"],
        rho_values=[0.0, 0.5],
    )
