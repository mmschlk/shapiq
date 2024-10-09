"""This script plots the explanations for a selection of games."""

import os
from itertools import product

import pandas as pd

from framework_utils import get_save_name

RESULTS_DIR = "framework_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":

    # params explanations
    feature_sets = [(0,), (1,), (2,), (3,)]
    feature_influences = ["pure", "partial", "full"]
    entities = ["individual", "joint", "interaction"]
    explanation_params = list(product(feature_sets, feature_influences, entities))

    # params games
    model_name = "lin_reg"  # lin_reg, xgb_reg, rnf_reg
    interaction_data = True  # False True
    rho_value = 0.5  # 0.0, 0.5, 0.9
    fanova_setting = "b"  # b c m
    n_instances = 100  # 100
    random_seed = 42  # 42
    num_samples = 10_000  # 10_000

    save_name = get_save_name(
        interaction_data=interaction_data,
        model_name=model_name,
        random_seed=random_seed,
        num_samples=num_samples,
        rho=rho_value,
        fanova=fanova_setting,
        instance_id=0,
    )

    # get game
    # save results
    results_df = pd.read_csv(os.path.join(RESULTS_DIR, save_name + ".csv"))
