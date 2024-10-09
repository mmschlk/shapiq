"""This script conducts a synthetic experiment for the unified framework."""

import numpy as np

from framework_explanation_game import LocalExplanationGame
from framework_utils import get_storage_dir, get_synth_data_and_model

if __name__ == "__main__":

    # Experiment settings
    RANDOM_SEED = 42
    model_name = "lin_reg"
    num_samples = 100_000
    interaction_data = False

    # Explanation Settings
    sample_size = 100_000
    n_instances = 1
    fanova_settings = ["m"]

    # get the directory for saving
    game_storage_path = get_storage_dir(model_name)

    x_data, y_data, model = get_synth_data_and_model(
        model_name=model_name,
        random_seed=RANDOM_SEED,
        rho=0.0,
        interaction_data=interaction_data,
        num_samples=num_samples,
    )
    print("Mean", np.mean(y_data))

    local_game = LocalExplanationGame(
        fanova="m",
        model=model,
        x_data=x_data,
        x_explain=x_data[0],
        loss_function=None,
        sample_size=sample_size,
        random_seed=RANDOM_SEED,
        normalize=False,
        verbose=True,
    )
    print("Val_empty", local_game(np.array([[False, False, False, False]])))
