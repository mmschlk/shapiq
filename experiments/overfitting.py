from __future__ import annotations

import multiprocessing as mp

import numpy as np

from shapiq.approximator.regression.shapleygax import (
    ExplanationBasisGenerator,
    ShapleyGAX,
)
from shapiq.benchmark import load_games_from_configuration

if __name__ == "__main__":
    # this code runs interventional treeshap from the shap package for ground truth and uses the TreeSHAPInterventionalXAI class
    ID_EXPLANATIONS = range(
        10
    )  # range(10,30) # ids of test instances to explain, can be used to compute new ids
    RANDOM_STATE = 40  # random state for the games
    # ID_CONFIG_APPROXIMATORS = 40  # PAIRING=False, REPLACEMENT=True
    ID_CONFIG_APPROXIMATORS = 39  # PAIRING_False, REPLACEMENT=False
    # ID_CONFIG_APPROXIMATORS = 38  # PAIRING=True, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 37  # PAIRING=True, REPLACEMENT=False

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

    RUN_GROUND_TRUTH = False
    RUN_APPROXIMATION = True

    GAME_IDENTIFIERS = [
        "SentimentIMDBDistilBERT14",
        "ResNet18w14Superpixel",
        # "ViT3by3Patches",
        "ViT4by4Patches",
    ]  # , "ImageClassifierLocalXAI"]
    # game_identifier = "SOUM"
    config_id = 1
    n_games = 10

    GAMES = {}

    for game_identifier in GAME_IDENTIFIERS:
        if game_identifier == "SentimentIMDBDistilBERT14":
            sentiment_analysis = load_games_from_configuration(
                game_class="SentimentAnalysisLocalXAI",
                n_player_id=0,
                config_id=config_id,
                n_games=n_games,
            )
            GAMES[game_identifier] = sentiment_analysis
        if game_identifier == "ResNet18w14Superpixel":
            image_classifier = load_games_from_configuration(
                game_class="ImageClassifierLocalXAI",
                n_player_id=0,
                config_id=config_id,
                n_games=n_games,
            )
            GAMES[game_identifier] = image_classifier
        if game_identifier == "ViT3by3Patches":
            image_classifier = load_games_from_configuration(
                game_class="ImageClassifierLocalXAI",
                n_player_id=1,
                config_id=config_id,
                n_games=n_games,
            )
            GAMES[game_identifier] = image_classifier
        if game_identifier == "ViT4by4Patches":
            image_classifier = load_games_from_configuration(
                game_class="ImageClassifierLocalXAI",
                n_player_id=2,
                config_id=config_id,
                n_games=n_games,
            )
            GAMES[game_identifier] = image_classifier

    if RUN_GROUND_TRUTH:
        # Compute the ground truth values for the games
        for game_identifier, game in GAMES.items():
            for id_explain, game_instance in enumerate(game):
                game_id = game_identifier + "_" + str(config_id)
                save_path = (
                    "ground_truth/overfitting/"
                    + game_id
                    + "_"
                    + str(RANDOM_STATE)
                    + "_"
                    + str(id_explain)
                    + "_exact_values.json"
                )
                ground_truth = game_instance.exact_values(index="SV", order=1)
                ground_truth.save(save_path)
                print(f"Exact: {ground_truth} saved to {save_path}")

    MAX_BUDGET = 5000

    def explain_instance(args):
        game_id, id_explain, game_instance = args

        # approximators = get_approximators(
        #     APPROXIMATORS, game_instance.n_players, RANDOM_STATE, PAIRING, REPLACEMENT
        # )
        approximators = []
        n_players = game_instance.n_players
        explanation_basis = ExplanationBasisGenerator(N=set(range(n_players)))
        # initialize the weights for KernelSHAP
        kernelshap_weights = np.zeros(n_players + 1)
        for size in range(1, n_players):
            kernelshap_weights[size] = 1 / (size * (n_players - size))

        # initialize the weights for LeverageSHAP
        leverage_weights_1 = np.ones(n_players + 1)
        k_exceed = False
        for k in range(1, n_players + 1):
            kadd = explanation_basis.generate_kadd_explanation_basis(max_order=k)
            if k_exceed:
                continue
            if len(kadd) > MAX_BUDGET:
                k_exceed = True
            # ShapleyGAX with k-add explanation basis
            shapley_gax_kadd = ShapleyGAX(
                n=n_players,
                explanation_basis=kadd,
                random_state=RANDOM_STATE,
                sampling_weights=kernelshap_weights,
                pairing_trick=PAIRING,
                replacement=REPLACEMENT,
            )
            shapley_gax_kadd.name = "ShapleyGAX-" + str(k) + "ADD"
            approximators.append(shapley_gax_kadd)
            shapley_gax_kadd_lev1 = ShapleyGAX(
                n=n_players,
                explanation_basis=kadd,
                random_state=RANDOM_STATE,
                sampling_weights=leverage_weights_1,
                pairing_trick=PAIRING,
                replacement=REPLACEMENT,
            )
            shapley_gax_kadd_lev1.name = "ShapleyGAX-" + str(k) + "ADD-Lev1"
            approximators.append(shapley_gax_kadd_lev1)

        for approximator in approximators:
            print(
                "Computing approximations for",
                approximator.name,
                "on game",
                game_id,
                "explanation id",
                id_explain,
            )
            shap_approx = approximator.approximate(budget=MAX_BUDGET, game=game_instance)
            save_path = (
                "approximations/overfitting/"
                + game_id
                + "_"
                + str(ID_CONFIG_APPROXIMATORS)
                + "_"
                + str(id_explain)
                + "_"
                + approximator.name
                + "_"
                + str(MAX_BUDGET)
                + ".json"
            )
            shap_approx.save(save_path)

    if RUN_APPROXIMATION:
        N_JOBS = 5
        # Compute the ground truth values for the games
        for game_identifier, game in GAMES.items():
            game_id = game_identifier + "_" + str(config_id)
            args_list = [
                (game_id, id_explain, game_instance)
                for id_explain, game_instance in enumerate(game)
            ]
            with mp.Pool() as pool:
                pool.map(explain_instance, args_list)
