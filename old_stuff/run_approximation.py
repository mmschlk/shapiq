"""This script evaluates the performance of several approximation methods."""

from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapiq import KernelSHAP, PermutationSamplingSV
from shapiq.approximator.regression.polyshap import (
    ShapleyGAX,
    ExplanationBasisGenerator,
)

from shapiq.benchmark import load_games_from_configuration, run_benchmark

from scipy.special import binom

from shapiq.utils.empirical_leverage_scores import get_leverage_scores


if __name__ == "__main__":
    RANDOM_STATE = 40
    GAMES_IDENTIFIER = [
        "AdultCensusLocalXAI"
    ]  # ["SentimentAnalysisLocalXAI","ImageClassifierLocalXAI"]

    for game_identifier in GAMES_IDENTIFIER:
        print(f"Running benchmark for {game_identifier}...")
        # read these values from the configuration file / or the printed benchmark configurations
        # game_identifier = "SentimentAnalysisLocalXAI" #AdultCensusLocalXAI CaliforniaHousingLocalXAI SentimentAnalysisLocalXAI ImageClassifierLocalXAI SynthDataTreeSHAPIQXAI
        config_id = 1

        if game_identifier == "ImageClassifierLocalXAI":
            N_PLAYER_IDS = [0, 2]
        else:
            N_PLAYER_IDS = [0]

        if game_identifier == "AdultCensusLocalXAI":
            config_id = 3
        else:
            config_id = 1

        for n_player_id in N_PLAYER_IDS:
            # load the game files from disk / or download
            games = load_games_from_configuration(
                game_class=game_identifier,
                n_player_id=n_player_id,
                config_id=config_id,
                n_games=10,
            )
            games = list(games)  # convert to list (the generator is consumed)
            n_players = games[0].n_players

            leverage_weights_1 = np.ones(n_players + 1)

            lev_scores_2 = get_leverage_scores(n_players, 2)
            leverage_weights_2 = np.zeros(n_players + 1)
            for size, score in lev_scores_2.items():
                leverage_weights_2[size] = binom(n_players, size) * score

            # budget steps from 500 to min(2**n_players, 20000) in 10 steps
            budget_steps = np.linspace(
                min(500, 0.1 * 2**n_players), min(2**n_players, 20000), 10
            ).astype(int)

            kernel_shap = KernelSHAP(n=n_players, random_state=RANDOM_STATE)
            kernel_shap.name = "KernelSHAP"
            leverage_shap = KernelSHAP(
                n=n_players,
                random_state=RANDOM_STATE,
                sampling_weights=leverage_weights_1,
            )
            leverage_shap.name = "LeverageSHAP"

            explanation_basis = ExplanationBasisGenerator(N=set(range(n_players)))

            kadd = explanation_basis.generate_kadd_explanation_basis(max_order=2)
            shapley_gax_kadd = ShapleyGAX(
                n=n_players, explanation_basis=kadd, random_state=RANDOM_STATE
            )
            shapley_gax_kadd.name = "ShapleyGAX-2ADD"
            shapley_gax_kadd_lev = ShapleyGAX(
                n=n_players,
                explanation_basis=kadd,
                random_state=RANDOM_STATE,
                sampling_weights=leverage_weights_1,
            )
            shapley_gax_kadd_lev.name = "ShapleyGAX-2ADD-Leverage1"
            shapley_gax_kadd_lev2 = ShapleyGAX(
                n=n_players,
                explanation_basis=kadd,
                random_state=RANDOM_STATE,
                sampling_weights=leverage_weights_2,
            )
            shapley_gax_kadd_lev2.name = "ShapleyGAX-2ADD-Leverage2"

            approximators = [
                kernel_shap,
                leverage_shap,
                shapley_gax_kadd,
                shapley_gax_kadd_lev,
                shapley_gax_kadd_lev2,
                PermutationSamplingSV(n=n_players, random_state=RANDOM_STATE),
            ]

            # get the index and order
            index = "SV"
            order = 1
            save_path = (
                "data/" + game_identifier + "_" + str(n_player_id) + "_results.json"
            )

            results = run_benchmark(
                index=index,
                order=order,
                games=games,
                approximators=approximators,
                save_path=save_path,
                # alternatively, you can set also max_budget (e.g. 10_000) and budget_step to 0.05 (in percentage of max_budget)
                budget_steps=budget_steps,
                rerun_if_exists=False,  # if True, the benchmark will rerun the approximators even if the results file exists
                n_jobs=12,  # number of parallel jobs
            )
