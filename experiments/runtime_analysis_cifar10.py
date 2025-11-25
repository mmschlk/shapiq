from __future__ import annotations

import multiprocessing as mp

import numpy as np
from init_approximator import get_approximators

from experiments.cifar10 import LocalXAICIFAR10Game

import pandas as pd

if __name__ == "__main__":
    """
    This script runs selected explanation games using path-dependent feature perturbation
    and collects runtime data for different approximators and budgets.
    """
    ID_EXPLANATIONS = range(
        10
    )  # range(10,30) # ids of test instances to explain, can be used to compute new ids
    RANDOM_STATE = 40  # random state for the games
    # ID_CONFIG_APPROXIMATORS = 40, PAIRING=False, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 39, PAIRING_False, REPLACEMENT=False
    # ID_CONFIG_APPROXIMATORS = 38, PAIRING=True, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 37, PAIRING=True, REPLACEMENT=False
    ID_CONFIG_APPROXIMATORS = 37  # used for different approximator configurations
    if ID_CONFIG_APPROXIMATORS == 40:
        REPLACEMENT = True
        PAIRING = False
    if ID_CONFIG_APPROXIMATORS == 39:
        REPLACEMENT = False
        PAIRING = False
    if ID_CONFIG_APPROXIMATORS == 38:
        REPLACEMENT = True
        PAIRING = True
    if ID_CONFIG_APPROXIMATORS == 37:
        REPLACEMENT = False
        PAIRING = True

    PRINT_PERFORMANCE = False
    RUN_GROUND_TRUTH = False
    RUN_APPROXIMATION = True

    APPROXIMATORS = [
        "RegressionMSR",
        "LeverageSHAP",
        "PolySHAP-2ADD",
        "PolySHAP-3ADD",
        "PolySHAP-4ADD",
    ]

    MAX_BUDGET = 20000
    N_BUDGET_STEPS = 10

    def explain_instance(args):
        game_instance, game_id, id_explain, runtime_df = args
        approximators = get_approximators(
            APPROXIMATORS, game_instance.n_players, RANDOM_STATE, PAIRING, REPLACEMENT
        )
        min_budget = game_instance.n_players + 1
        max_budget = min(2**game_instance.n_players, MAX_BUDGET)
        budget_range = np.logspace(
            np.log10(min_budget), np.log10(max_budget), N_BUDGET_STEPS
        ).astype(int)
        # print(budget_range)
        for approximator in approximators:
            if game_instance.n_players > 30 and approximator.name in [
                "PolySHAP-3ADD",
                "PolySHAP-3ADD-10%",
                "PolySHAP-3ADD-20%",
                "PolySHAP-3ADD-50%",
                "PolySHAP-3ADD-75%",
            ]:
                continue
            if game_instance.n_players > 20 and approximator.name in [
                "PolySHAP-4ADD",
            ]:
                continue
            print(
                "Computing approximations for",
                approximator.name,
                "on game",
                game_id,
                "explanation id",
                id_explain,
            )
            for budget in budget_range:
                try:
                    shap_approx = approximator.approximate(
                        budget=budget, game=game_instance
                    )
                except Exception as e:
                    import traceback
                    print(f"Couldn't compute (approximator={getattr(approximator, 'name', None)}, budget={budget}): {e}")
                    traceback.print_exc()
                    continue
                df = pd.DataFrame([approximator.runtime_last_approximate_run])
                df["game_id"] = game_id
                df["id_explain"] = id_explain
                df["approximator"] = approximator.name
                df["budget"] = budget
                df["n_players"] = game_instance.n_players
                df["id_config_approximator"] = ID_CONFIG_APPROXIMATORS
                runtime_df = pd.concat([runtime_df, df])
                runtime_df.to_csv(f"runtime_analysis_cifar10.csv")
        return runtime_df

    if RUN_APPROXIMATION:
        runtime_df = pd.DataFrame()
        # add CIFAR10 using ViT inference
        for id_explain in ID_EXPLANATIONS:
            cifar10_game = LocalXAICIFAR10Game(id_explain=id_explain, random_state=RANDOM_STATE, use_model=True)
            game_id = "CIFAR10_1"
            runtime_df = explain_instance((cifar10_game, game_id, id_explain, runtime_df))