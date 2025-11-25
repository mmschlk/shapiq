from __future__ import annotations

import multiprocessing as mp

import numpy as np
from init_approximator import get_approximators

from shapiq.games.benchmark.local_xai import AdultCensus, BikeSharing, CaliforniaHousing
from shapiq.games.benchmark.local_xai.benchmark_tabular import (
    NHANESI,
    BreastCancer,
    CommunitiesAndCrime,
    Corrgroups60,
    ForestFires,
    IndependentLinear60,
    RealEstate,
    WineQuality,
)
from shapiq.games.benchmark.treeshapiq_xai import TreeSHAPIQXAI

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

    # run the benchmark for the games
    GAMES = [
        CaliforniaHousing(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        RealEstate(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        BreastCancer(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        IndependentLinear60(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
    ]

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
        game_id, id_explain, runtime_df = args
        tree_game = TREE_GAMES[id_explain]
        approximators = get_approximators(
            APPROXIMATORS, game.n_players, RANDOM_STATE, PAIRING, REPLACEMENT
        )
        min_budget = tree_game.n_players + 1
        max_budget = min(2**game.n_players, MAX_BUDGET)
        budget_range = np.logspace(
            np.log10(min_budget), np.log10(max_budget), N_BUDGET_STEPS
        ).astype(int)
        # print(budget_range)
        for approximator in approximators:
            if tree_game.n_players > 30 and approximator.name in [
                "PolySHAP-3ADD",
                "PolySHAP-3ADD-10%",
                "PolySHAP-3ADD-20%",
                "PolySHAP-3ADD-50%",
                "PolySHAP-3ADD-75%",
            ]:
                continue
            if tree_game.n_players > 20 and approximator.name in [
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
                        budget=budget, game=tree_game
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
                df["n_players"] = tree_game.n_players
                df["id_config_approximator"] = ID_CONFIG_APPROXIMATORS
                runtime_df = pd.concat([runtime_df, df])
                runtime_df.to_csv(f"runtime_analysis.csv")
        return runtime_df

    if RUN_APPROXIMATION:
        runtime_df = pd.DataFrame()
        for game in GAMES:
            game_id = game.setup.dataset_name + "_" + game.setup.model_name
            TREE_GAMES = []
            for id_explain in ID_EXPLANATIONS:
                x_explain = game.setup.x_test[id_explain, :]
                tree_game = TreeSHAPIQXAI(x_explain, game.setup.model, verbose=False)
                TREE_GAMES.append(tree_game)
                runtime_df = explain_instance((game_id, id_explain, runtime_df))
            print(
                "Tree games initialized for",
                game.setup.dataset_name,
                game.setup.model_name,
            )
