from __future__ import annotations

import multiprocessing as mp

import numpy as np
from init_approximator import get_approximators

from shapiq import InteractionValues
from shapiq.games.benchmark.local_xai import AdultCensus, BikeSharing, CaliforniaHousing
from shapiq.games.benchmark.local_xai.benchmark_tabular import (
    NHANESI,
    BreastCancer,
    CommunitiesAndCrime,
    Corrgroups60,
    IndependentLinear60,
    WineQuality,
)
from shapiq.games.benchmark.treeshap_interventional_xai import TreeSHAPInterventionalXAI

if __name__ == "__main__":
    # this code runs interventional treeshap from the shap package for ground truth and uses the TreeSHAPInterventionalXAI class
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
    RUN_GROUND_TRUTH = True
    RUN_APPROXIMATION = True

    # run the benchmark for the games
    GAMES = [
        CaliforniaHousing(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        # CaliforniaHousing(
        #     model_name="gradient_boosting",
        #     imputer="baseline",
        #     random_state=RANDOM_STATE,
        # ),
        WineQuality(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        # WineQuality(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        BikeSharing(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        # BikeSharing(
        #     model_name="gradient_boosting",
        #     imputer="baseline",
        #     random_state=RANDOM_STATE,
        # ),
        # ForestFires(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        # ForestFires(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        AdultCensus(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        # AdultCensus(
        #     model_name="gradient_boosting",
        #     imputer="baseline",
        #     random_state=RANDOM_STATE,
        # ),
        # RealEstate(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        # RealEstate(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        BreastCancer(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        # BreastCancer(
        #     model_name="gradient_boosting",
        #     imputer="baseline",
        #     random_state=RANDOM_STATE,
        # ),
        IndependentLinear60(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        # IndependentLinear60(
        #     model_name="gradient_boosting",
        #     imputer="baseline",
        #     random_state=RANDOM_STATE,
        # ),
        Corrgroups60(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        # Corrgroups60(
        #     model_name="gradient_boosting",
        #     imputer="baseline",
        #     random_state=RANDOM_STATE,
        # ),
        NHANESI(model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE),
        # NHANESI(model_name="gradient_boosting", imputer="baseline", random_state=RANDOM_STATE),
        CommunitiesAndCrime(
            model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
        ),
        # CommunitiesAndCrime(
        #     model_name="gradient_boosting",
        #     imputer="baseline",
        #     random_state=RANDOM_STATE,
        # ),
        # SentimentAnalysis(),
        # ImageClassifier()
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
                background_data = np.mean(game.setup.x_train, axis=0, keepdims=True)
                if (
                    game.setup.dataset_type == "classification"
                    and game.setup.model_name != "gradient_boosting"
                ):
                    class_to_explain = int(
                        np.argmax(game.setup.predict_function(x_explain.reshape(1, -1)))
                    )
                else:
                    class_to_explain = None
                tree_game = TreeSHAPInterventionalXAI(
                    x_explain,
                    game.setup.model,
                    class_label=class_to_explain,
                    verbose=False,
                    feature_perturbation="interventional",
                    background_data=background_data,
                )
                save_path = (
                    "ground_truth/baseline/"
                    + game_id
                    + "_"
                    + str(RANDOM_STATE)
                    + "_"
                    + str(id_explain)
                    + "_exact_values.json"
                )
                shap_ground_truth_numpy = tree_game.exact_values(index="SV", order=1)
                shap_ground_truth_values = np.zeros(game.n_players + 1)
                shap_ground_truth_values[0] = tree_game.empty_value
                shap_ground_truth_values[1:] = shap_ground_truth_numpy
                # shap_ground_truth = InteractionValues(shap_ground_truth_numpy[:,class_to_explain], index="SV",estimated=False,max_order=1,n_players=game.n_players,min_order=0,baseline_value=tree_game.empty_value[class_to_explain])
                # else:
                shap_ground_truth = InteractionValues(
                    shap_ground_truth_values,
                    index="SV",
                    max_order=1,
                    n_players=game.n_players,
                    min_order=0,
                    baseline_value=tree_game.empty_value,
                    estimated=False,
                )
                shap_ground_truth.save(save_path)

                # compute shap values with ExactComputer
                # from shapiq import ExactComputer
                # exact_computer = ExactComputer(n_players=game.n_players, game=tree_game)
                # exhaustive_shap_values = exact_computer(index="SV", order=1)
                print(f"Exact: {shap_ground_truth.values} saved to {save_path}")

    APPROXIMATORS = [
        "PermutationSampling",
        "KernelSHAP",
        "LeverageSHAP",
        # "PolySHAP-2ADD",
        # "PolySHAP-3ADD",
        # "PolySHAP-4ADD",
        # "PolySHAP-2ADD-10%",
        # "PolySHAP-2ADD-20%",
        # "PolySHAP-2ADD-50%",
        # "PolySHAP-2ADD-75%",
        # "PolySHAP-3ADD-10%",
        # "PolySHAP-3ADD-20%",
        # "PolySHAP-3ADD-50%",
        # "PolySHAP-3ADD-75%",
    ]

    MAX_BUDGET = 20000
    N_BUDGET_STEPS = 10

    def explain_instance(args):
        game_id, id_explain = args
        tree_game = TREE_GAMES[id_explain]
        approximators = get_approximators(
            APPROXIMATORS, tree_game.n_players, RANDOM_STATE, PAIRING, REPLACEMENT
        )
        min_budget = min(50, 2**tree_game.n_players / 10)
        max_budget = min(2**game.n_players, MAX_BUDGET)
        budget_range = np.logspace(
            np.log10(min_budget), np.log10(max_budget), N_BUDGET_STEPS
        ).astype(int)
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
                "PolySHAP-4ADD",
                "PolySHAP-3ADD-10%",
                "PolySHAP-3ADD-20%",
                "PolySHAP-3ADD-50%",
                "PolySHAP-3ADD-75%",
            ]:
                continue
            for budget in budget_range:
                shap_approx = approximator.approximate(budget=budget, game=tree_game)
                save_path = (
                    "approximations/baseline/"
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

    if RUN_APPROXIMATION:
        N_JOBS = 5
        for game in GAMES:
            game_id = game.setup.dataset_name + "_" + game.setup.model_name
            TREE_GAMES = []
            for id_explain in ID_EXPLANATIONS:
                x_explain = game.setup.x_test[id_explain, :]
                background_data = np.mean(game.setup.x_train, axis=0, keepdims=True)
                if (
                    game.setup.dataset_type == "classification"
                    and game.setup.model_name != "gradient_boosting"
                ):
                    class_to_explain = int(
                        np.argmax(game.setup.predict_function(x_explain.reshape(1, -1)))
                    )
                    tree_game = TreeSHAPInterventionalXAI(
                        x_explain,
                        game.setup.model,
                        class_label=class_to_explain,
                        verbose=False,
                        feature_perturbation="interventional",
                        background_data=background_data,
                    )
                else:
                    tree_game = TreeSHAPInterventionalXAI(
                        x_explain,
                        game.setup.model,
                        verbose=False,
                        feature_perturbation="interventional",
                        background_data=background_data,
                    )
                TREE_GAMES.append(tree_game)
            print(
                "Tree games initialized for",
                game.setup.dataset_name,
                game.setup.model_name,
            )
            args_list = [(game_id, id_explain) for id_explain in ID_EXPLANATIONS]
            with mp.Pool() as pool:
                pool.map(explain_instance, args_list)
