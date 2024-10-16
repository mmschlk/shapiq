"""This script computes the explanations for a selection of games."""

import os

import tqdm
from matplotlib import pyplot as plt

from framework_explanation_game import (
    LocalExplanationGame,
    MultiDataExplanationGame,
    loss_binary_crossentropy,
    mse,
)
from framework_explanations import compute_explanation_int_val
from framework_si_graph import si_graph_plot
from framework_utils import (
    get_ml_data,
    get_save_name_ml,
    get_storage_dir,
)
from shapiq import Game, InteractionValues, powerset

RESULTS_DIR = "framework_results_ml"
os.makedirs(RESULTS_DIR, exist_ok=True)

PLOT_DIR = "framework_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def abbreviate_feature_names(features: list[str]) -> list[str]:
    abbrev = []
    for name in features:
        if len(name) == 3:
            abbrev.append(name)
            continue
        upper_letters = [c for c in name if c.isupper()]
        if len(upper_letters) == 1:
            abbrev.append(name[:2])
        elif len(upper_letters) == 0:
            abbrev.append(name[:2])
        else:
            abbrev.append("".join(upper_letters))
    return abbrev


def plot_bar_plot(exp: InteractionValues, feature_mapping: dict[int, str]) -> None:

    fig, ax = plt.subplots()

    order_one = exp.get_n_order(min_order=1, order=1)

    # get the feature names
    features = [feature_mapping[i] for i in range(order_one.n_players)]
    for i in range(order_one.n_players):
        ax.bar(i, order_one[(i,)], label=features[i])

    ax.set_xticks(range(order_one.n_players))
    ax.set_xticklabels(features, rotation=45)
    ax.set_ylabel("Effect")
    title = f"{fanova_setting}-FANOVA and {feature_influence} influence"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # interesting settings -------------------------------------------------------------------------

    # 1)
    # data: titanic, model_name: xgb, local_explanation_instance: 12, plot_local: True
    # -> an older women who survived because of the higher class and high fare
    # -> model also predicts it correctly

    # 2)
    # data: bike, model_name: nn, plot_global: True, n_instances: 100
    # -> the most important features is the hour

    # settings -------------------------------------------------------------------------------------

    # save the results
    save_results = True
    plot_bar = False

    # main settings
    plot_local = False
    plot_global = True
    plot_sensitivity = False
    assert sum([plot_local, plot_global, plot_sensitivity]) == 1, "Only one plot can be selected."

    # compute settings
    re_compute = False
    do_k_fold = False

    # data and model settings
    data_name = "titanic"  # "california", "bike", or "titanic"
    model_name = "xgb"  # "xgb", "rnf", or "nn"

    # framework explanations
    feature_influence = "pure"  # "pure", "partial", "full"
    fanova_setting = "m"  # "c", "b", "m"
    entity = "interaction"  # "individual", "joint", "interaction"
    order = None  # 1, 2, 3, ...

    # random seed
    random_seed = 42
    sample_size = 512

    # global/sensitivity settings
    n_instances = 100

    # local settings
    local_explanation_instance = 12

    # get the data and model -----------------------------------------------------------------------
    model, x_data, y_data, x_train, x_test, y_train, y_test, feature_names = get_ml_data(
        model_name=model_name, random_seed=random_seed, data_name=data_name, do_k_fold=do_k_fold
    )
    feature_names_abbrev = abbreviate_feature_names(features=feature_names)
    print(f"Feature names: {feature_names}")
    print(f"Abbreviated feature names: {feature_names_abbrev}")
    if order is None:
        order = len(feature_names)

    # get the save name ----------------------------------------------------------------------------
    plot_save_path = "_".join(
        [data_name, model_name, feature_influence, fanova_setting, entity, str(order)]
    )
    if plot_local:
        plot_save_path += f"_local_{local_explanation_instance}"
    elif plot_global:
        plot_save_path += f"_global_{n_instances}"
    elif plot_sensitivity:
        plot_save_path += f"_sensitivity_{n_instances}"
    plot_save_path += ".pdf"
    plot_save_path = os.path.join(PLOT_DIR, plot_save_path)

    # plot a local explanation ---------------------------------------------------------------------
    if plot_local:
        x_explain = x_test[local_explanation_instance]
        y_explain = y_test[local_explanation_instance]
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
        explanation = compute_explanation_int_val(
            game=local_game,
            entity_type=entity,
            influence=feature_influence,
            explanation_order=order,
        )
        print("Explaining Instance:", local_explanation_instance)
        print("Instance", x_explain)
        print("Label", y_explain)
        print("Model prediction:", model.predict(x_explain.reshape(1, -1)))

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
                if re_compute:
                    raise FileNotFoundError
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
            loss_function = mse if data_name != "titanic" else loss_binary_crossentropy
            game = MultiDataExplanationGame(
                local_games=local_games,
                y_targets=y_test[:n_instances],
                sensitivity_game=False,
                loss_function=loss_function,
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
    print("Sum of all effects:", sum(explanation.values) + explanation.baseline_value)
    int_values = explanation.get_n_order(min_order=1, order=order)
    print(int_values)
    label_mapping = {i: name for i, name in enumerate(feature_names_abbrev)}
    si_graph_nodes = list(powerset(range(int_values.n_players), min_size=2, max_size=2))
    si_graph_plot(
        int_values,
        graph=si_graph_nodes,
        draw_original_edges=False,
        circular_layout=True,
        label_mapping=label_mapping,
        compactness=1_000_000_000_000,
        size_factor=3,
        node_size_scaling=1.5,
        # draw_threshold=0.001*max(abs(int_values.values)),
        n_interactions=100,
        node_area_scaling=False,
    )
    plt.tight_layout()
    if save_results:
        plt.savefig(plot_save_path)
    plt.show()

    # plot the bar plot
    if plot_bar:
        plot_bar_plot(exp=int_values, feature_mapping=label_mapping)
