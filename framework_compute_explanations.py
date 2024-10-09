"""This script computes the explanations for a selection of games."""

import os
from itertools import product

import numpy as np
import pandas as pd
import tqdm

from framework_explanations import compute_explanation, compute_explanation_with_mi
from framework_utils import get_save_name, load_local_games

RESULTS_DIR = "framework_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _update_results(
    _results: list,
    _explanation: dict[tuple[int, ...], float],
    _game_id: int,
    _feature_influence: str,
    _entity: str,
    _x_explain: np.ndarray,
    _y_explain: float,
) -> None:
    _y_explain = _y_explain
    for _feature_set, _exp_val in _explanation.items():
        _x_val = float(_x_explain[_feature_set])
        if len(_feature_set) == 1:
            _feature_set = _feature_set[0]
        else:
            _feature_set = tuple(_feature_set)
        _results.append(
            {
                "game_id": _game_id,
                "feature_set": _feature_set,
                "feature_influence": _feature_influence,
                "entity": _entity,
                "explanation": _exp_val,
                "feature_value": _x_val,
                "y_explain": _y_explain,
                "explanation/feature_value": _exp_val / _x_val if _x_val != 0 else 0,
            }
        )


def _compare_explanations(
    _explanation_mi: dict, _explanation: dict, print_error: bool = False
) -> None:
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

    check_similar = True

    # params explanations
    feature_sets = [(0,), (1,), (2,), (3,)]
    feature_influences = ["pure", "partial", "full"]
    entities = ["individual"]
    explanation_params = list(product(feature_influences, entities))

    # params games
    model_name = "lin_reg"  # lin_reg, xgb_reg, rnf_reg
    n_instances = 1  # 100
    random_seed = 42  # 42
    num_samples = 10_000  # 10_000
    sample_size = 128  # 1_000
    interaction_datas = [False]  # False True
    rho_values = [0.0, 0.5, 0.9]  # 0.0, 0.5, 0.9
    fanova_settings = ["c"]  # b c m
    setting_params = list(product(interaction_datas, rho_values, fanova_settings))

    for interaction_data, rho_value, fanova_setting in setting_params:
        save_name = get_save_name(
            interaction_data=interaction_data,
            model_name=model_name,
            random_seed=random_seed,
            num_samples=num_samples,
            rho=rho_value,
            fanova=fanova_setting,
            sample_size=sample_size,
            instance_id=0,
        )

        # get game
        games, x_explain, y_explain = load_local_games(
            model_name=model_name,
            interaction_data=interaction_data,
            rho_value=rho_value,
            fanova_setting=fanova_setting,
            n_instances=n_instances,
            random_seed=random_seed,
            num_samples=num_samples,
            sample_size=sample_size,
        )

        pbar = tqdm.tqdm(total=len(explanation_params) * len(games) * 2)

        # compute explanations
        results, results_mi = [], []
        for feature_influence, entity in explanation_params:
            for game_id, game in enumerate(games):
                # compute explanation via MIs
                explanation_mi = compute_explanation_with_mi(
                    game=game,
                    feature_sets=feature_sets,
                    feature_influence=feature_influence,
                    entity=entity,
                )
                pbar.update(1)
                _update_results(
                    _results=results_mi,
                    _explanation=explanation_mi,
                    _game_id=game_id,
                    _feature_influence=feature_influence,
                    _entity=entity,
                    _x_explain=x_explain[game_id],
                    _y_explain=y_explain[game_id],
                )

                # compute explanation via shapiq
                explanation = compute_explanation(
                    game=game,
                    feature_sets=feature_sets,
                    feature_influence=feature_influence,
                    entity=entity,
                )
                pbar.update(1)
                _update_results(
                    _results=results,
                    _explanation=explanation,
                    _game_id=game_id,
                    _feature_influence=feature_influence,
                    _entity=entity,
                    _x_explain=x_explain[game_id],
                    _y_explain=y_explain[game_id],
                )

                # check if explanations are somewhat similar
                if check_similar:
                    _compare_explanations(_explanation_mi=explanation_mi, _explanation=explanation)

        # save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(RESULTS_DIR, f"{save_name}_explanations.csv"), index=False)
