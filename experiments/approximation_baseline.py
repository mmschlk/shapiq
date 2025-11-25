from __future__ import annotations
import numpy as np

from experiments.cifar10 import LocalXAICIFAR10Game
from init_approximator import get_approximators

import os
import multiprocessing as mp
from shapiq.benchmark import load_games_from_configuration

if __name__ == "__main__":
    """
    This script runs selected approximation algorithms on explanation games that use baseline
    imputatation, which were pre-computed in the shapiq library. The ground truth values
    are computed using exhaustive evaluation. Approximations are stored in
    /approximations/exhaustive/ and ground truth values in /ground_truth/exhaustive/.
    """
    RECOMPUTE = False # If False, then will skip approximation if file exists
    RANDOM_STATE = 40  # random state for the games
    # ID_CONFIG_APPROXIMATORS = 40  # PAIRING=False, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 39  # PAIRING=False, REPLACEMENT=False
    # ID_CONFIG_APPROXIMATORS = 38  # PAIRING=True, REPLACEMENT=True
    ID_CONFIG_APPROXIMATORS = 37  # PAIRING=True, REPLACEMENT=False

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
        "CIFAR10",
        "SentimentIMDBDistilBERT14",
        "ResNet18w14Superpixel",
        "ViT3by3Patches",
        "ViT4by4Patches",
    ]  # , "ImageClassifierLocalXAI"]
    # game_identifier = "SOUM"
    config_id = 1
    n_games = 30 #only 30 available

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
        if game_identifier == "CIFAR10":
            cifar10_games = []
            for id_explain in range(n_games):
                cifar10_game = LocalXAICIFAR10Game(id_explain=id_explain, random_state=RANDOM_STATE)
                save_path = f"cifar10_precomputed/cifar10_{id_explain}_game_values.npz"
                cifar10_game.load_values(save_path)
                cifar10_games.append(cifar10_game)
            GAMES[game_identifier] = cifar10_games


    if RUN_GROUND_TRUTH:
        # Compute the ground truth values for the games
        for game_identifier, game in GAMES.items():
            for id_explain, game_instance in enumerate(game):
                game_id = game_identifier + "_" + str(config_id)
                save_path = (
                    "ground_truth/exhaustive/"
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

    APPROXIMATORS = [
        "MSR",
        "SVARM",
        "RegressionMSR",
        "PermutationSampling",
        "LeverageSHAP",
        "PolySHAP-2ADD",
        "PolySHAP-3ADD",
        "PolySHAP-4ADD",
        "PolySHAP-2ADD-50%",
        "PolySHAP-3ADD-50%",
        "PolySHAP-3ADD-dlog(d)",
    ]

    MAX_BUDGET = 20000
    N_BUDGET_STEPS = 10

    def explain_instance(args):
        game_id, id_explain, game_instance = args
        approximator_list = get_approximators(
            APPROXIMATORS,
            game_instance.n_players,
            RANDOM_STATE,
            PAIRING,
            REPLACEMENT,
        )
        min_budget = game_instance.n_players + 1
        max_budget = min(2**game_instance.n_players, MAX_BUDGET)
        budget_range = np.round(np.logspace(
            np.log10(min_budget), np.log10(max_budget), N_BUDGET_STEPS
        )).astype(int)
        for approximator in approximator_list:
            print(
                "Computing approximations for",
                approximator.name,
                "on game",
                game_id,
                "explanation id",
                id_explain,
            )
            for budget in budget_range:
                save_path = (
                    "approximations/exhaustive/"
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
                if (not RECOMPUTE) and os.path.exists(save_path):
                    print(f"Skipping existing file {save_path}")
                    continue
                if hasattr(approximator, "n_variables") and approximator.n_variables > budget:
                    print(f"Skipping budget {budget} due to too many variables {approximator.n_variables}")
                    continue
                shap_approx = approximator.approximate(
                    budget=budget, game=game_instance
                )
                shap_approx.save(save_path)

    if RUN_APPROXIMATION:
        N_JOBS = 15
        # Compute the ground truth values for the games
        for game_identifier, game in GAMES.items():
            game_id = game_identifier + "_" + str(config_id)
            # for id_explain, game_instance in enumerate(game):
            #     explain_instance((game_id, id_explain, game_instance))
            args_list = [
                (game_id, id_explain, game_instance)
                for id_explain, game_instance in enumerate(game)
            ]
            # use for parallelization, requires a lot of RAM for high-dimensional games
            with mp.Pool() as pool:
                pool.map(explain_instance, args_list)
