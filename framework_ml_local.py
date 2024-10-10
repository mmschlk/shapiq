"""This script conducts a synthetic experiment for the unified framework."""

import os
import warnings

import tqdm

from framework_explanation_game import LocalExplanationGame
from framework_utils import get_california_data_and_model

if __name__ == "__main__":

    # disable all warnings
    warnings.filterwarnings("ignore")
    PBAR = True
    VERBOSE_GAME = not PBAR

    # Experiment settings
    RANDOM_SEED = 42
    model_name = "xgb_reg"

    # Explanation Settings
    sample_size = 512
    n_instances = 2
    fanova_settings = ["c", "m", "b"]

    # get the directory for saving
    game_storage_path = os.path.join("game_storage", "local", "california", model_name)
    os.makedirs(game_storage_path, exist_ok=True)

    model, x_data, y_data, x_train, x_test, y_train, y_test = get_california_data_and_model(
        model_name=model_name, random_seed=RANDOM_SEED
    )
    for fanova in fanova_settings:
        if PBAR:
            n_games = n_instances
            pbar = tqdm.tqdm(total=n_games)
        else:
            pbar = None
        for instance_id in range(n_instances):
            name = "_".join([model_name, str(RANDOM_SEED), fanova, str(instance_id)])
            save_path = os.path.join(game_storage_path, name)
            if os.path.exists(save_path + ".npz"):
                if PBAR:
                    pbar.update(1)
                continue
            local_game = LocalExplanationGame(
                fanova=fanova,
                model=model,
                x_data=x_test[0:500],
                x_explain=x_test[instance_id],
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
