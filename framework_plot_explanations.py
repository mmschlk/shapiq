"""This script plots the explanations for a selection of games."""

import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from framework_utils import get_save_name

RESULTS_DIR = "framework_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def quick_boxplot(data: pd.DataFrame, _feature_influence: str, _entity: str) -> None:
    """Draws a quick boxplot of the data."""
    fig, ax = plt.subplots()
    data_selection = data[
        (data["feature_influence"] == _feature_influence) & (data["entity"] == _entity)
    ]
    data_selection.boxplot(column="explanation", by="feature_set", ax=ax)
    plt.title(
        f"Feature Influence: {_feature_influence}, Entity: {_entity}, FANOVA {fanova_setting}"
    )
    plt.show()


def draw_bar_plot(
    dfs: list[pd.DataFrame],
    feature_sets: list[tuple[int, ...]],
    feature_influences: list[str],
    entities: list[str],
    fanova_settings: list[str],
) -> None:
    """Draws a bar plot of the data.

    The bar plot consists of multiple groups of bars, where each bar is a feature set. Each group
    of bars corresponds to a different feature influence + entity + fanova combination. Each group
    is of a different color.
    """
    pass


if __name__ == "__main__":

    # plot params
    plot_mi_explanations = False

    # params explanations
    feature_sets = [(0,), (1,), (2,), (3,)]
    feature_influences = ["full"]
    entities = ["individual"]
    explanation_params = list(product(feature_influences, entities))

    # params games
    model_name = "lin_reg"  # lin_reg, xgb_reg, rnf_reg
    interaction_data = True  # False True
    rho_value = 0.9  # 0, 0.5, 0.9
    fanova_setting = "c"  # b c m
    n_instances = 100  # 100
    random_seed = 42  # 42
    num_samples = 10_000  # 10_000

    # get the save name
    save_name = get_save_name(
        interaction_data=interaction_data,
        model_name=model_name,
        random_seed=random_seed,
        num_samples=num_samples,
        rho=rho_value,
        fanova=fanova_setting,
        instance_id=0,
    )
    mi_save_id = ""
    if plot_mi_explanations:
        mi_save_id = "_mi"
    save_path = os.path.join(RESULTS_DIR, f"{save_name}_explanations{mi_save_id}.csv")

    # load the explanations
    explanations_df = pd.read_csv(save_path)

    # plot the explanations
    for feature_influence, entity in explanation_params:
        quick_boxplot(explanations_df, feature_influence, entity)
