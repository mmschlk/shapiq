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

if __name__ == "__main__":
    """
    This script runs selected approximation algorithms on explanation games that use path-dependent
    feature perturbation. A random forest was trained and the ground truth values are computed via
    TreeSHAP. Approximations are stored in /approximations/pathdependent/ and ground truth values in
    /ground_truth/pathdependent/.
    """
    ID_EXPLANATIONS = range(
        10
    )  # range(10,30) # ids of test instances to explain, can be used to compute new ids
    RANDOM_STATE = 40  # random state for the games
    # ID_CONFIG_APPROXIMATORS = 40, PAIRING=False, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 39, PAIRING_False, REPLACEMENT=False
    # ID_CONFIG_APPROXIMATORS = 38, PAIRING=True, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 37, PAIRING=True, REPLACEMENT=False
    ID_CONFIG_APPROXIMATORS = 39  # used for different approximator configurations
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
        BikeSharing(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        ForestFires(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        AdultCensus(
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
        Corrgroups60(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        NHANESI(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        CommunitiesAndCrime(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
    ]

    if PRINT_PERFORMANCE:
        for game in GAMES:
            print(game.setup.dataset_name, game.setup.model_name, game.n_players)
            game.setup.print_train_performance()

    if RUN_GROUND_TRUTH:
        # Compute the ground truth values for the games
        for game in GAMES:
            game_id = game.setup.dataset_name + "_" + game.setup.model_name
            print(game.setup.dataset_name, game.setup.model_name)
            for id_explain in ID_EXPLANATIONS:
                x_explain = game.setup.x_test[id_explain, :]
                tree_game = TreeSHAPIQXAI(x_explain, game.setup.model, verbose=False)
                save_path = (
                    "ground_truth/pathdependent/"
                    + game_id
                    + "_"
                    + str(RANDOM_STATE)
                    + "_"
                    + str(id_explain)
                    + "_exact_values.json"
                )
                shap_ground_truth = tree_game.exact_values(index="SV", order=1)
                shap_ground_truth.save(save_path)
                print(f"Exact: {shap_ground_truth.values} saved to {save_path}")

    APPROXIMATORS = [
        # "MSR",
        # "SVARM",
        "RegressionMSR",
        "PermutationSampling",
        # "KernelSHAP",
        "LeverageSHAP",
        "PolySHAP-2ADD",
        "PolySHAP-3ADD",
        "PolySHAP-4ADD",
        # "PolySHAP-2ADD-10%",
        # "PolySHAP-2ADD-20%",
        "PolySHAP-2ADD-50%",
        # "PolySHAP-2ADD-75%",
        # "PolySHAP-3ADD-10%",
        # "PolySHAP-3ADD-20%",
        "PolySHAP-3ADD-50%",
        # "PolySHAP-3ADD-75%",
    ]

    MAX_BUDGET = 20000
    N_BUDGET_STEPS = 10

    def explain_instance(args):
        game_id, id_explain = args
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
            print(
                "Computing approximations for",
                approximator.name,
                "on game",
                game_id,
                "explanation id",
                id_explain,
            )
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
            for budget in budget_range:
                try:
                    shap_approx = approximator.approximate(
                        budget=budget, game=tree_game
                    )
                    save_path = (
                        "approximations/pathdependent/"
                        + game_id
                        + "_"
                        + str(ID_CONFIG_APPROXIMATORS)
                        + "_"
                        + str(id_explain)
                        + "_"
                        + approximator.name
                        + "_"
                        + str(budget)
                        + ".json"
                    )
                    shap_approx.save(save_path)
                except:
                    print("Couldn't compute")
                    continue

    if RUN_APPROXIMATION:
        N_JOBS = 5
        for game in GAMES:
            game_id = game.setup.dataset_name + "_" + game.setup.model_name
            TREE_GAMES = []
            for id_explain in ID_EXPLANATIONS:
                x_explain = game.setup.x_test[id_explain, :]
                tree_game = TreeSHAPIQXAI(x_explain, game.setup.model, verbose=False)
                TREE_GAMES.append(tree_game)
            print(
                "Tree games initialized for",
                game.setup.dataset_name,
                game.setup.model_name,
            )
            args_list = [(game_id, id_explain) for id_explain in ID_EXPLANATIONS]
            for id_explain in ID_EXPLANATIONS:
                explain_instance((game_id, id_explain))
            # use instead for parallelization, does not work well with RegressionMSR
            # with mp.Pool() as pool:
            #     pool.map(explain_instance, args_list)
