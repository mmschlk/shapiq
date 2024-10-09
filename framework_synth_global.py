"""This script conducts a synthetic experiment for the global and sensitivity games."""

import os
import warnings

import numpy as np

from framework_explanation_game import MultiDataExplanationGame, loss_mse
from framework_utils import (
    get_save_name,
    get_storage_dir,
    get_synth_data_and_model,
    load_local_games,
)

if __name__ == "__main__":

    # disable all warnings
    warnings.filterwarnings("ignore")
    VERBOSE_GAME = True

    # Experiment settings
    RANDOM_SEED = 42
    model_name = "xgb_reg"
    num_samples = 10_000
    rho_values = [0, 0.5, 0.9]
    interaction_data = False

    # Explanation Settings
    sample_size = 128
    n_instances = 100
    fanova_settings = ["b", "c", "m"]
    game_type = "global"  # global, sensitivity

    game_storage_path = get_storage_dir(model_name, game_type)

    _, y_data, _ = get_synth_data_and_model(
        model_name="lin_reg",
        random_seed=RANDOM_SEED,
        rho=0,
        interaction_data=interaction_data,
        num_samples=num_samples,
    )

    for rho_value in rho_values:
        for fanova in fanova_settings:
            name = get_save_name(
                interaction_data, model_name, RANDOM_SEED, num_samples, rho_value, fanova, 0
            )
            save_path = os.path.join(game_storage_path, name)
            if os.path.exists(save_path + ".npz"):
                continue
            local_games, x_explain, y_explain = load_local_games(
                model_name=model_name,
                interaction_data=interaction_data,
                rho_value=rho_value,
                fanova_setting=fanova,
                n_instances=n_instances,
                random_seed=RANDOM_SEED,
                num_samples=num_samples,
            )
            y_explain = np.asarray(y_explain)
            assert y_explain.shape[0] == len(local_games)

            global_game = MultiDataExplanationGame(
                local_games=local_games, y_targets=y_explain, loss_function=loss_mse
            )
            global_game.precompute()

            global_game.save_values()
