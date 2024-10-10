"""This script conducts a synthetic experiment for the unified framework."""

import os
import warnings

import numpy as np
import tqdm
from matplotlib import pyplot as plt

from framework_explanation_game import LocalExplanationGame, MultiDataExplanationGame, loss_mse
from framework_utils import get_california_data_and_model
from shapiq import Game


def _load_local_game(_fanova: str) -> list[Game]:
    all_files = os.listdir(os.path.join("game_storage", "local", "california", model_name))
    # check weather _fanova is in the file name
    all_files = [f for f in all_files if f"_{_fanova}_" in f]
    all_files = [f for f in all_files if "global" not in f]
    games = []
    for file in all_files:
        path = os.path.join("game_storage", "local", "california", model_name, file)
        game = Game(path_to_values=path, normalize=False, verbose=False)
        games.append(game)
    return games


if __name__ == "__main__":

    # disable all warnings
    warnings.filterwarnings("ignore")
    PBAR = True
    VERBOSE_GAME = not PBAR

    plot_global = False
    plot_local = True

    # Experiment settings
    RANDOM_SEED = 42
    # model_name = "xgb_reg"
    model_name = "neural_net"

    # Explanation Settings
    sample_size = 512
    n_instances = 20
    fanova_settings = ["m"]

    # get the directory for saving
    game_storage_path = os.path.join("game_storage", "local", "california", model_name)
    os.makedirs(game_storage_path, exist_ok=True)

    model, x_data, y_data, x_train, x_test, y_train, y_test, feature_names = (
        get_california_data_and_model(model_name=model_name, random_seed=RANDOM_SEED)
    )

    explanation_instances = []
    for i in range(n_instances):
        explanation_instances.append((x_test[i], y_test[i]))

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

    # compute global game --------------------------------------------------------------------------
    global_name = "_".join([model_name, str(RANDOM_SEED), "global"])
    for fanova in fanova_settings:
        local_games = _load_local_game(fanova)
        local_games = local_games[:n_instances]
        assert len(local_games) == len(
            explanation_instances
        ), f"{len(local_games)} != {len(explanation_instances)}"

        y_targets = np.array([y for x, y in explanation_instances])
        print(y_targets.shape)

        global_game = MultiDataExplanationGame(
            local_games=local_games,
            y_targets=y_targets,
            loss_function=loss_mse,
        )
        global_game.precompute()
        global_game.save_values(os.path.join(game_storage_path, f"{global_name}_{fanova}"))

    # visualize the games --------------------------------------------------------------------------

    from networkx import circular_layout

    from framework_si_graph import si_graph_plot
    from shapiq import ExactComputer, powerset

    # abbreviate feature_names by taking all upper case letters
    # if a word has only one upper letter then the second letter is also taken
    feature_names_abbrev = []
    for name in feature_names:
        upper_letters = [c for c in name if c.isupper()]
        if len(upper_letters) == 1:
            feature_names_abbrev.append(name[:2])
        else:
            feature_names_abbrev.append("".join(upper_letters))

    # visualize global game with si plots ----------------------------------------------------------
    if plot_global:
        for fanova in fanova_settings:
            global_game = Game(
                path_to_values=os.path.join(game_storage_path, f"{global_name}_{fanova}.npz"),
                normalize=False,
                verbose=False,
            )
            computer = ExactComputer(global_game.n_players, global_game)
            si_values = computer(index="Moebius", order=global_game.n_players)
            si_graph_nodes = list(powerset(range(si_values.n_players), min_size=2, max_size=2))
            print(si_values)
            # pos = circular_layout(si_graph_nodes)
            label_mapping = {i: name for i, name in enumerate(feature_names_abbrev)}
            si_graph_plot(
                si_values,
                graph=si_graph_nodes,
                draw_original_edges=False,
                circular_layout=True,
                label_mapping=label_mapping,
                node_size_scaling=1,
                compactness=100,
                n_interactions=100,
            )
            model_str = "XGBoost" if model_name == "xgb_reg" else "Random Forest"
            plt.title(f"{fanova}-fANOVA SAGE Interactions\n{model_str} for California Housing Data")
            plt.tight_layout()
            plt.show()

    # plot a local game ----------------------------------------------------------------------------
    if plot_local:
        # for instance_id_to_plot in range(79, 80):
        for instance_id_to_plot in range(0, 10):
            print(f"Plotting instance {instance_id_to_plot}")
            print(f"Model: {model_name}")
            print("Instance:", x_test[instance_id_to_plot])
            print("True value:", y_test[instance_id_to_plot])
            print("Predicted value:", model.predict(x_test[instance_id_to_plot].reshape(1, -1)))
            min_size, max_size = 0, 0
            interactions = []
            for fanova in ["m"]:
                # for influence in ["partial"]:
                for influence in ["partial", "pure"]:
                    path = os.path.join(
                        game_storage_path,
                        f"{model_name}_{RANDOM_SEED}_{fanova}_{instance_id_to_plot}.npz",
                    )
                    local_game = Game(path_to_values=path, normalize=False)
                    order = 2
                    computer = ExactComputer(local_game.n_players, local_game)
                    if influence == "pure":
                        int_values = computer(index="Moebius", order=order)
                    elif influence == "partial":
                        int_values = computer(index="k-SII", order=order)
                    else:
                        int_values = computer(index="Co-Moebius", order=order)
                    int_values = int_values.get_n_order(min_order=1, order=order)
                    interactions.append(int_values)
                    min_val = np.min(int_values.values)
                    max_val = np.max(int_values.values)
                    min_size = min(min_size, min_val)
                    max_size = max(max_size, max_val)
                    print(int_values)

            for int_values in interactions:
                if int_values.index == "Moebius":
                    influence = "pure"
                elif int_values.index == "Co-Moebius":
                    influence = "full"
                else:
                    influence = "partial"
                si_graph_nodes = list(powerset(range(int_values.n_players), min_size=2, max_size=2))
                pos = circular_layout(si_graph_nodes)
                label_mapping = {i: name for i, name in enumerate(feature_names_abbrev)}
                # increase font size
                plt.rcParams.update({"font.size": 17})
                si_graph_plot(
                    int_values,
                    graph=si_graph_nodes,
                    draw_original_edges=False,
                    circular_layout=True,
                    label_mapping=label_mapping,
                    n_interactions=100,
                    compactness=100_000_00000,
                    size_factor=3,
                    node_size_scaling=1.3,
                    # min_max_interactions=(min_size, max_size),
                    node_area_scaling=False,
                )
                plt.tight_layout()
                plt.savefig(f"local_game_{instance_id_to_plot}_{influence}_{fanova}.pdf")
                plt.title(f"{instance_id_to_plot} {influence} {fanova}")
                plt.show()
