"""This script plots the explanations for a selection of games."""

import os
from itertools import product

# import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from framework_utils import get_save_name

RESULTS_DIR = "framework_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_explanation_data(
    num_samples: int = 10_000,
    model_name: str = "lin_reg",
    rho_values: list[float] = (0.0, 0.5, 0.9),
    sample_sizes: list[int] = (512,),
    interaction_data: bool = True,
    ones_instances: bool = (True,),
    n_instances: list[int] = (1,),
    random_seeds: int = 30,
    only_load: bool = False,
) -> pd.DataFrame:
    """Loads the explanation data from disk."""

    if only_load:
        try:
            data_all_df = pd.read_csv(os.path.join(RESULTS_DIR, "all_explanations.csv"))
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
    data_all_df.to_csv(os.path.join(RESULTS_DIR, "all_explanations.csv"), index=False)


if __name__ == "__main__":

    # plot params
    plot_mi_explanations = False

    # params explanations
    feature_sets = [(0,), (1,), (2,), (3,), (1, 2), (1, 2, 3)]
    feature_influences = ["pure", "partial", "full"]
    fanova_settings = ["c", "b", "m"]
    entities = ["individual", "joint", "interaction"]

    # load the data
    data_all_df = load_explanation_data(only_load=True)
