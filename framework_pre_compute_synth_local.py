"""This script conducts a synthetic experiment for the unified framework."""

import os
import warnings
from itertools import product

import numpy as np
import tqdm

from framework_explanation_game import LocalExplanationGame
from framework_utils import (
    SynthConditionalSampler,
    get_save_name_synth,
    get_storage_dir,
    get_synth_data_and_model,
)

if __name__ == "__main__":

    # disable all warnings
    warnings.filterwarnings("ignore")
    PBAR = True
    VERBOSE_GAME = not PBAR
    RUN_IF_EXISTS = True

    # random seed
    random_seeds = list(range(1))

    # Experiment settings
    model_names = ["lin_reg"]
    num_samples = 10_000
    rho_values = [0.0, 0.5, 0.9]
    interaction_datas = [None, "linear-interaction", "non-linear-interaction"]
    synth_settings = list(product(model_names, rho_values, interaction_datas))

    # Explanation Settings
    sample_sizes = [512]
    n_instances_list = [50]
    fanova_settings = ["c", "b", "m"]
    ones_instances = [True]
    game_settings = list(product(sample_sizes, n_instances_list, fanova_settings, ones_instances))

    if PBAR:
        n_games = len(random_seeds) * len(synth_settings) * len(game_settings) - 1
        n_games *= sum(n_instances_list)
        pbar = tqdm.tqdm(total=n_games)
    else:
        pbar = None

    # iterate over all settings
    for random_seed in random_seeds:
        for model_name, rho, interaction_data in synth_settings:
            # get the directory for saving
            game_storage_path = get_storage_dir(model_name)
            for sample_size, n_instances, fanova, ones_instance in game_settings:
                x_data, y_data, model = get_synth_data_and_model(
                    random_seed=random_seed,
                    rho=rho,
                    interaction_data=interaction_data,
                    num_samples=num_samples,
                )
                local_games = []
                for instance_id in range(n_instances):
                    data_name = "synthetic_ones" if ones_instance else "synthetic"
                    name = get_save_name_synth(
                        interaction_data=interaction_data,
                        model_name=model_name,
                        random_seed=random_seed,
                        num_samples=num_samples,
                        rho=rho,
                        fanova=fanova,
                        instance_id=instance_id,
                        sample_size=sample_size,
                        data_name=data_name,
                    )
                    save_path = os.path.join(game_storage_path, name)
                    if not RUN_IF_EXISTS and os.path.exists(save_path + ".npz"):
                        if PBAR:
                            pbar.update(1)
                        continue
                    cond_sampler = None
                    if fanova == "c":
                        cond_sampler = SynthConditionalSampler(
                            n_features=x_data.shape[1],
                            sample_size=sample_size,
                            random_seed=random_seed,
                            rho=rho,
                        )
                    x_explain = np.ones(x_data.shape[1]) if ones_instance else x_data[instance_id]
                    local_game = LocalExplanationGame(
                        fanova=fanova,
                        model=model,
                        x_data=x_data,
                        x_explain=x_explain,
                        loss_function=None,
                        sample_size=sample_size,
                        random_seed=random_seed,
                        normalize=False,
                        verbose=VERBOSE_GAME,
                        cond_sampler=cond_sampler,
                    )
                    # pre-compute the game values
                    local_game.precompute()
                    local_game.save_values(save_path)
                    if PBAR:
                        pbar.update(1)
