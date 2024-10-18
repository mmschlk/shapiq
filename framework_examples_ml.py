"""This script computes the explanations for a selection of games."""

import os

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

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
from shapiq.datasets import load_titanic
from shapiq.plot._config import get_color

RESULTS_DIR = "framework_results_ml"
os.makedirs(RESULTS_DIR, exist_ok=True)

PLOT_DIR = "framework_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

CP = [
    "#00b4d8",
    "#ef27a6",
    "#ff6f00",
    "#ffbe0b",
    "#ef27a6",
    "#7DCE82",
    "#00b4d8",
    "#ef27a6",
    "#ff6f00",
    "#ffbe0b",
    "#ef27a6",
]


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


def plot_bar_plot(
    exp: InteractionValues, feature_mapping: dict[int, str], color_feature: bool = False
) -> None:

    figsize = (5.2, 4)

    fig, axis = plt.subplots(figsize=figsize)
    order_one = exp.get_n_order(min_order=1, order=1)

    # get the feature names
    features = [feature_mapping[i] for i in range(order_one.n_players)]
    x_pos, bar_width, bar_padding, x_positions = 0, 0.1, 0.05, []
    for f_id, feature_set in enumerate(features):
        y_value = exp[(f_id,)]
        if color_feature:
            color = CP[f_id]
        else:
            color = get_color(y_value)
        axis.bar(x=x_pos, height=y_value, width=bar_width, color=color)
        x_positions.append(x_pos)
        x_pos += bar_width + bar_padding

    axis.set_xticks(x_positions)
    axis.set_xticklabels(features)
    axis.set_ylabel(y_label)
    axis.set_xlabel("Feature")
    axis.axhline(0, color="gray", linestyle="--", alpha=0.75, linewidth=1, zorder=1)
    # capitalize first letter
    axis.set_title(title, fontsize=title_fontsize)
    if data_name == "titanic" and plot_sensitivity:
        axis.set_ylim(0, 10)
    elif data_name == "titanic" and plot_local and local_explanation_instance == 134:
        axis.set_ylim(-2, 3.9)
    elif data_name == "california" and plot_global:
        axis.set_ylim(-1.5, 2.2)
    elif data_name == "bike" and plot_global:
        axis.set_ylim(-0.1, 1.3)
    plt.tight_layout()


if __name__ == "__main__":

    # settings for the plots in the paper ----------------------------------------------------------

    # 1) Local Titanic:
    # data: titanic, model_name: xgb, plot_local: True, local_explanation_instance: 134

    # 2) Global Bike
    # data: bike, model_name: nn, plot_global: True, n_instances: 100

    # 3) Global California
    # data: california, model_name: nn, plot_global: True, n_instances: 500

    # 4) Sensitivity Titanic
    # data: titanic, model_name: xgb, plot_sensitivity: True, n_instances: 500

    # settings -------------------------------------------------------------------------------------

    # save the results
    save_results = True
    plot_title = True

    # main settings specify one plot type to be True
    plot_local = True
    plot_global = False
    plot_sensitivity = False
    assert sum([plot_local, plot_global, plot_sensitivity]) == 1, "Only one plot can be selected."

    # compute settings
    re_compute = False  # to always recompute the games (takes about 30 mins for the global games)

    # data and model settings
    data_name = "titanic"  # "california", "bike", or "titanic"
    model_name = "xgb"  # "xgb", "rnf", or "nn"

    # framework explanations
    feature_influence = "pure"  # "pure", "partial", "full"
    fanova_setting = "m"  # "c", "b", "m"
    entity = "interaction"  # "individual", "joint", "interaction"
    order = 1  # 1, 2, 3, ...

    # random seed
    random_seed = 42
    sample_size = 512

    # global/sensitivity settings
    n_instances = 100

    # local settings
    local_explanation_instance = 134
    # 26 good

    # get the data and model -----------------------------------------------------------------------
    model, x_data, y_data, x_train, x_test, y_train, y_test, feature_names = get_ml_data(
        model_name=model_name, random_seed=random_seed, data_name=data_name, do_k_fold=False
    )
    feature_names_abbrev = abbreviate_feature_names(features=feature_names)
    print(f"Feature names: {feature_names}")
    print(f"Abbreviated feature names: {feature_names_abbrev}")
    if order is None:
        order = len(feature_names)

    # print the model and data information ---------------------------------------------------------
    if data_name == "titanic":
        print("Titanic data")
        if plot_local:
            x_raw, y_raw = load_titanic(to_numpy=False, pre_processing=False)
            _, x_test_raw, _, _ = train_test_split(
                x_raw, y_raw, train_size=0.7, shuffle=True, random_state=random_seed
            )
            print("Local Explanation Instance")
            print("Columns", x_raw.columns)
            print("Instance", x_test_raw.iloc[local_explanation_instance])
        correct_instances_survived, correct_instances_died = [], []
        for x_idx in range(len(x_test)):
            x_instance = x_test[x_idx]
            y_instance = y_test[x_idx]
            y_pred = model.predict(x_instance.reshape(1, -1))
            if y_pred == y_instance:
                if y_instance == 1:
                    correct_instances_survived.append(x_instance)
                    if x_instance[1] == 1:
                        print(f"{x_idx}: {x_instance}")
                else:
                    correct_instances_died.append(x_instance)
        print("Survival Rate of Whole Data:", np.mean(y_data))
        x_embarked = x_data[:, 4]
        for embarked_value in np.unique(x_embarked):
            survived = y_data[x_embarked == embarked_value]
            print(f"Embarked: {embarked_value}, Survived: {np.mean(survived)}")
        x_sex = x_data[:, 1]
        for sex_value in np.unique(x_sex):
            survived = y_data[x_sex == sex_value]
            print(f"Sex: {sex_value}, Survived: {np.mean(survived)}")

    # create a plot title --------------------------------------------------------------------------
    influence_str = feature_influence.capitalize()
    data_names = {"california": "California housing", "bike": "Bike sharing", "titanic": "Titanic"}
    title = f"{data_names[data_name]} ({fanova_setting}-fANOVA, {influence_str}, order: {order})"
    title_fontsize = 13
    label_fontsize = 11
    plt.rcParams.update({"font.size": label_fontsize})
    plt.rcParams.update({"axes.titlesize": title_fontsize})
    y_label = "Importance" if plot_global else "Effect"
    y_label = "Sensitivity" if plot_sensitivity else y_label

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

    if order != 1:
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
        if plot_title:
            plt.title(title)
        plt.tight_layout()
        if save_results:
            plt.savefig(plot_save_path + ".pdf")
        plt.show()
    else:
        plot_bar_plot(exp=int_values, feature_mapping=label_mapping)
        if save_results:
            plt.savefig(plot_save_path + "_bar.pdf")
        plt.show()
