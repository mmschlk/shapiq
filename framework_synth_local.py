"""This script conducts a synthetic experiment for the unified framework."""

import os
import warnings

import tqdm

from framework_explanation_game import LocalExplanationGame
from framework_utils import get_save_name, get_storage_dir, get_synth_data_and_model

if __name__ == "__main__":

    # disable all warnings
    warnings.filterwarnings("ignore")
    PBAR = True
    VERBOSE_GAME = not PBAR

    # Experiment settings
    RANDOM_SEED = 42
    model_name = "lin_reg"
    num_samples = 10_000
    rho_values = [0.0, 0.5, 0.9]
    interaction_data = False

    # Explanation Settings
    sample_size = 128
    n_instances = 1
    fanova_settings = ["b", "m"]

    # get the directory for saving
    game_storage_path = get_storage_dir(model_name)

    if PBAR:
        n_games = len(rho_values) * n_instances * len(fanova_settings)
        pbar = tqdm.tqdm(total=n_games)
    else:
        pbar = None

    for i, rho in enumerate(rho_values):
        x_data, y_data, model = get_synth_data_and_model(
            model_name=model_name,
            random_seed=RANDOM_SEED,
            rho=rho,
            interaction_data=interaction_data,
            num_samples=num_samples,
        )
        for instance_id in range(n_instances):
            for fanova in fanova_settings:
                name = get_save_name(
                    interaction_data, model_name, RANDOM_SEED, num_samples, rho, fanova, instance_id
                )
                save_path = os.path.join(game_storage_path, name)
                if os.path.exists(save_path + ".npz"):
                    if PBAR:
                        pbar.update(1)
                    continue
                local_game = LocalExplanationGame(
                    fanova=fanova,
                    model=model,
                    x_data=x_data,
                    x_explain=x_data[instance_id],
                    loss_function=None,
                    sample_size=sample_size,
                    random_seed=RANDOM_SEED,
                    normalize=False,
                    verbose=VERBOSE_GAME,
                )
                # pre-compute the game values
                local_game.precompute()
                local_game.save_values(save_path)
                if PBAR:
                    pbar.update(1)
