"""This module contains utility functions for the experiments."""

import os
from warnings import warn

import shapiq

RANDOM_SEED = 42


def make_file_paths(file_name: str, budget: int, experiment: str) -> tuple[str, str]:
    """Returns the file paths for stored game values and potential interaction values."""
    file_name = file_name.split(".")[0]  # remove any file extension if present
    file_name = f"{file_name}_budget={budget}_seed={RANDOM_SEED}.npz"
    experiment_dir = os.path.join("results", experiment)
    game_values_dir = os.path.join(experiment_dir, "game_values")
    os.makedirs(game_values_dir, exist_ok=True)
    interaction_values_dir = os.path.join(experiment_dir, "interaction_values")
    os.makedirs(interaction_values_dir, exist_ok=True)
    file_path_game = os.path.join(game_values_dir, file_name)
    file_path_interaction = os.path.join(interaction_values_dir, file_name)
    return file_path_game, file_path_interaction


def load_game_from_file(
    file_name: str, budget: int = 1_000_000, experiment: str = "vit"
) -> shapiq.Game:
    """Loads a precomputed game from a file."""
    file_path, _ = make_file_paths(file_name, budget, experiment)
    return shapiq.Game(path_to_values=file_path)


def load_gt_values(
    file_name: str, budget: int = 1_000_000, experiment: str = "vit"
) -> shapiq.InteractionValues:
    """Loads the ground truth Shapley values from a file."""
    _, file_path = make_file_paths(file_name, budget, experiment)
    return shapiq.InteractionValues.load_interaction_values(path=file_path)


def pre_compute_model_values(
    image_name: str,
    budget: int = 1_000_000,
    experiment: str = "vit",
    recompute_if_exists: bool = False,
    **kwargs,
) -> None:
    """Precomputes the model values for the given game and budget."""

    if budget != 1_000_000:
        warn(f"The budget is not 1_000_000 (default) but {budget}. ")

    # get the file paths
    file_path_game, file_path_interaction = make_file_paths(image_name, budget, experiment)

    # check if the values are already computed
    if os.path.exists(file_path_game) and os.path.exists(file_path_interaction):
        print(f"Values for file {image_name} with budget {budget} already exist.")
        if recompute_if_exists:
            print("Recomputing values.")
        else:
            return

    if experiment == "vit":
        from experiment_vision_transformer import VisionTransformerGame

        image_path = os.path.join("images", image_name)
        game = VisionTransformerGame(x_explain_path=image_path, verbose=True, **kwargs)
    else:
        raise ValueError(f"Unknown experiment {experiment}")

    print(f"Precomputing values for {game.__class__.__name__} with budget {budget}")
    approximator = shapiq.KernelSHAP(n=game.n_players, random_state=RANDOM_SEED)
    sampler = approximator._sampler
    sampler.sample(sampling_budget=budget)
    game.precompute(coalitions=sampler.coalitions_matrix)
    game.save_values(path=file_path_game)
    print(f"Values saved to {file_path_game}")

    # compute the Shapley values for the game
    print(f"Computing Shapley values for {game.__class__.__name__} with budget {budget}")

    # initialize the approximator again
    loaded_game = load_game_from_file(image_name, budget, experiment)
    approximator = shapiq.KernelSHAP(n=loaded_game.n_players, random_state=RANDOM_SEED)
    sv = approximator.approximate(budget=budget, game=loaded_game)
    print(sv)
    print(f"Saving interaction values to {file_path_interaction}")
    sv.save(as_pickle=False, path=file_path_interaction)

    # test if interaction values are correct and can be loaded
    loaded_sv = load_gt_values(image_name, budget, experiment)
    print("Loaded interaction values:")
    print(loaded_sv)
