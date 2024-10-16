"""This script computes the explanations for a selection of games."""

import os

import tqdm
from matplotlib import pyplot as plt

from framework_explanation_game import LocalExplanationGame, MultiDataExplanationGame, loss_mse
from framework_explanations import compute_explanation_int_val
from framework_si_graph import si_graph_plot
from framework_utils import (
    get_ml_data,
    get_save_name_ml,
    get_storage_dir,
)
from shapiq import Game, powerset

RESULTS_DIR = "framework_results_ml"
os.makedirs(RESULTS_DIR, exist_ok=True)


def abbreviate_feature_names(
    features: list[str],
) -> list[str]:
    feature_names_abbrev = []
    for name in features:
        if len(name) == 3:
            feature_names_abbrev.append(name)
            continue
        upper_letters = [c for c in name if c.isupper()]
        if len(upper_letters) == 1:
            feature_names_abbrev.append(name[:2])
        elif len(upper_letters) == 0:
            feature_names_abbrev.append(name[:2])
        else:
            feature_names_abbrev.append("".join(upper_letters))
    return feature_names_abbrev


if __name__ == "__main__":

    check_similar = False

    data_name = "titanic"  # "california", "bike", or "titanic"
    model_name = "xgb"  # "xgb", "rnf", or "nn"

    # params explanations
    feature_influence = "partial"  # "pure", "partial", "full"
    fanova_setting = "m"  # "c", "b", "m"
    entity = "interaction"  # "individual", "joint", "interaction"

    # random seed
    random_seed = 42

    # game settings
    sample_size = 512
    n_instances = 4

    plot_local = True
    plot_global = False
    plot_sensitivity = False

    order = 2

    local_explanation_instance = 3

    # get a pbar
    model, x_data, y_data, x_train, x_test, y_train, y_test, feature_names = get_ml_data(
        model_name=model_name, random_seed=random_seed, data_name=data_name
    )
    feature_names_abbrev = abbreviate_feature_names(features=feature_names)
    print(f"Feature names: {feature_names}")
    print(f"Abbreviated feature names: {feature_names_abbrev}")

    # plot a local explanation ---------------------------------------------------------------------
    if plot_local:
        x_explain = x_test[local_explanation_instance]
        save_name = get_save_name_ml(
            model_name=model_name,
            fanova=fanova_setting,
            instance_id=local_explanation_instance,
            data_name=data_name,
            random_seed=random_seed,
        )
        game_storage_path = get_storage_dir(model_name, game_type=os.path.join("local", data_name))
        save_path = os.path.join(game_storage_path, save_name)
        try:
            local_game = Game(path_to_values=save_path, normalize=False)
        except FileNotFoundError:
            local_game = LocalExplanationGame(
                fanova=fanova_setting,
                model=model,
                x_data=x_test[0:sample_size],
                x_explain=x_explain,
                loss_function=None,
                sample_size=sample_size,
                random_seed=random_seed,
                normalize=False,
                verbose=False,
            )
            local_game.precompute()
            local_game.save_values(save_path)

        explanation = compute_explanation_int_val(
            game=local_game,
            entity_type=entity,
            influence=feature_influence,
            explanation_order=order,
        )
        print("Explaining Instance:", local_explanation_instance)
        print(x_explain)
        print(y_test[local_explanation_instance])

    # plot global oder sensitivity -----------------------------------------------------------------
    elif plot_sensitivity or plot_global:
        local_games = []
        pbar = tqdm.tqdm(total=n_instances)
        for instance_id in range(n_instances):
            x_explain = x_test[instance_id]
            save_name = get_save_name_ml(
                model_name=model_name,
                fanova=fanova_setting,
                instance_id=instance_id,
                data_name=data_name,
                random_seed=random_seed,
            )
            game_storage_path = get_storage_dir(
                model_name, game_type=os.path.join("local", data_name)
            )
            save_path = os.path.join(game_storage_path, save_name)
            try:
                local_game = Game(path_to_values=save_path, normalize=False)
            except FileNotFoundError:
                local_game = LocalExplanationGame(
                    fanova=fanova_setting,
                    model=model,
                    x_data=x_test[0:sample_size],
                    x_explain=x_explain,
                    loss_function=None,
                    sample_size=sample_size,
                    random_seed=random_seed,
                    normalize=False,
                    verbose=False,
                )
                local_game.precompute()
                local_game.save_values(save_path)
            local_games.append(local_game)
            pbar.update(1)

        # compute global explanations
        if plot_global:
            game = MultiDataExplanationGame(
                local_games=local_games,
                y_targets=y_test[:n_instances],
                sensitivity_game=False,
                loss_function=loss_mse,
            )
        else:
            game = MultiDataExplanationGame(
                local_games=local_games,
                y_targets=y_test[:n_instances],
                sensitivity_game=True,
            )
        game.precompute()
        explanation = compute_explanation_int_val(
            game=game,
            entity_type=entity,
            influence=feature_influence,
            explanation_order=order,
        )
    else:
        raise ValueError("No plot type selected.")

    # do the plotting for the explanations ---------------------------------------------------------

    # plot with an SI graph
    int_values = explanation.get_n_order(min_order=1, order=order)
    label_mapping = {i: name for i, name in enumerate(feature_names_abbrev)}
    si_graph_nodes = list(powerset(range(int_values.n_players), min_size=2, max_size=2))
    si_graph_plot(
        int_values,
        graph=si_graph_nodes,
        draw_original_edges=False,
        circular_layout=True,
        label_mapping=label_mapping,
        node_size_scaling=1.5,
        size_factor=2,
        compactness=100,
        n_interactions=100,
    )
    plt.tight_layout()
    plt.show()
