"""This script computes the explanations for a selection of games."""

import os
from itertools import product

import numpy as np
import pandas as pd
import tqdm

from framework_explanations import compute_explanation, compute_explanation_with_mi
from framework_utils import get_save_name_synth, load_local_games_synth, update_results

RESULTS_DIR = "framework_results_synth"
os.makedirs(RESULTS_DIR, exist_ok=True)


def compare_mi_to_shapiq(_explanation: dict, print_error: bool = False):
    """Compare the explanations from MI to the ones in shapiq"""
    # compute explanation via MIs
    _explanation_mi = compute_explanation_with_mi(
        game=game,
        feature_sets=feature_sets,
        feature_influence=feature_influence,
        entity=entity,
    )
    for _feature_set, _exp_val in _explanation.items():
        _exp_val_mi = _explanation_mi[_feature_set]
        similar = abs(_exp_val - _exp_val_mi) < 1e-3
        if not similar:
            msg = (
                f"Explanations are not similar: {(_exp_val, _exp_val_mi)} for feature set "
                f"{_feature_set}, feature influence {feature_influence}, entity {entity}, game id "
                f"{game_id}"
            )
            if print_error:
                print(msg)
            else:
                raise ValueError(msg)


if __name__ == "__main__":

    check_similar = False

    # params explanations
    feature_sets = [(0,), (1,), (2,), (3,), (1, 2), (1, 2, 3)]
    feature_influences = ["pure", "partial", "full"]
    fanova_settings = ["c", "b", "m"]
    entities = ["individual", "joint", "interaction"]
    explanation_params = list(product(feature_influences, fanova_settings, entities))

    # random seed
    random_seeds = list(range(30))

    # game settings
    model_names = ["lin_reg"]
    num_samples = 10_000
    rho_values = [0.0, 0.5, 0.9]
    interaction_datas = [None, "linear-interaction", "non-linear-interaction"]
    sample_sizes = [512]
    n_instances_list = [1]
    ones_instances = [True]
    game_settings = list(
        product(
            model_names,
            rho_values,
            interaction_datas,
            sample_sizes,
            n_instances_list,
            ones_instances,
        )
    )

    # get a pbar
    total = len(random_seeds) * (len(game_settings) - 1) * len(explanation_params)
    total *= sum(n_instances_list)
    pbar = tqdm.tqdm(total=total)

    for random_seed in random_seeds:
        for (
            model_name,
            rho_value,
            interaction_data,
            sample_size,
            n_instances,
            ones_instance,
        ) in game_settings:
            # get a save name for later
            data_name = "synthetic_ones" if ones_instance else "synthetic"
            save_name = get_save_name_synth(
                interaction_data=interaction_data,
                model_name=model_name,
                random_seed=random_seed,
                num_samples=num_samples,
                rho=rho_value,
                fanova="all",
                sample_size=sample_size,
                instance_id=0,
                data_name=data_name,
            )
            results = []
            for feature_influence, fanova_setting, entity in explanation_params:
                # get the games from disk
                games, x_explain, y_explain = load_local_games_synth(
                    model_name=model_name,
                    interaction_data=interaction_data,
                    rho_value=rho_value,
                    fanova_setting=fanova_setting,
                    n_instances=n_instances,
                    random_seed=random_seed,
                    num_samples=num_samples,
                    sample_size=sample_size,
                    data_name=data_name,
                )
                # compute explanations

                for game_id, game in enumerate(games):
                    # compute explanation via shapiq
                    explanation = compute_explanation(
                        game=game,
                        feature_sets=feature_sets,
                        feature_influence=feature_influence,
                        entity=entity,
                    )
                    x_explain = np.ones(4) if ones_instance else x_explain[game_id]
                    update_results(
                        _results=results,
                        _explanation=explanation,
                        _game_id=game_id,
                        _feature_influence=feature_influence,
                        _entity=entity,
                        _fanova_setting=fanova_setting,
                        _x_explain=x_explain,
                    )

                    # check if explanations are somewhat similar
                    if check_similar:
                        compare_mi_to_shapiq(_explanation=explanation)

                    pbar.update(1)

            # save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(
                os.path.join(RESULTS_DIR, f"{save_name}_explanations.csv"), index=False
            )
