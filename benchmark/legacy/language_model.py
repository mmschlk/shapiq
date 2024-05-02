"""This module contains the benchmark conducted on the language model game."""

import os
import sys
import glob

import numpy as np
from tqdm.auto import tqdm

import pandas as pd

from benchmark.legacy.legacy import OldLMGame, check_interaction
from benchmark.setup import get_interaction_approximator
from metrics import get_all_metrics
from precompute_lm import pre_compute_imdb

from shapiq.games import Game

if __name__ == "__main__":

    PRE_COMPUTE_IMDB = False  # use this file to pre-compute the games
    LOAD_OLD_GAME = True  # load the old game exactly as it was in KernelSHAPIQ
    TEST_BUDGETS = False  # test only a few budgets
    N_PLAYERS = 14  # number of players to load from
    N_GAMES = 1  # leave on 1 for debugging
    MAX_ORDER = 2
    INDEX = "SII"  # only SII or k-SII

    MAX_BUDGET = min(2**N_PLAYERS, 10_000)
    BUDGET_STEPS = [int(budget_step * MAX_BUDGET) for budget_step in np.arange(0.15, 1.05, 0.05)]
    if TEST_BUDGETS:
        BUDGET_STEPS = [4000, 5000]
    print("Budget steps: ", BUDGET_STEPS)

    if PRE_COMPUTE_IMDB:
        # pre-compute values for multiple games
        pre_compute_imdb(n_games=10, n_players=N_PLAYERS)
        print("Pre-computed games.")
        sys.exit(0)

    if LOAD_OLD_GAME:
        # pre-compute the old game
        path_to_values = "precomputed/OldSentimentAnalysis(Game)/14/game.npz"
        game = OldLMGame()
        game.save_values(path_to_values)

        new_game = Game(path_to_values=path_to_values)
        assert check_interaction(
            game_to_check=new_game, verbose=False
        ), "Game values are different."
        print("Pre-computed the old game.")
        games = [new_game]

    n_games_loaded = len(games)
    print("Loaded", n_games_loaded, "games.")

    # load benchmark approximators

    approximator_names = [
        "KernelSHAPIQ",
        "InconsistentKernelSHAPIQ",
        "SHAPIQ",
        "SVARMIQ",
        "PermutationSamplingSII",
    ]

    # print experiment parameters
    print(f"Experiment parameters:")
    print(f"Number of players: {N_PLAYERS}")
    print(f"Number of games: {n_games_loaded}")
    print(f"Interaction index: {INDEX}")
    print(f"Interaction order: {MAX_ORDER}")
    print(
        f"Budgets: {BUDGET_STEPS}, resulting in {sum(BUDGET_STEPS)} evaluations per approximator "
        f"per game."
    )
    print(f"Approximators: {approximator_names}")

    pbar = tqdm(total=sum(BUDGET_STEPS) * min(len(games), N_GAMES) * len(approximator_names))

    results = []
    for budget in BUDGET_STEPS:
        for i, game in enumerate(games):
            if i >= N_GAMES:
                break
            # print("Benchmarking", game.game_name, "with", game.n_players, "players.")
            exact_values = game.exact_values(index=INDEX, order=MAX_ORDER)
            for approximator_name in approximator_names:
                approximator = get_interaction_approximator(approximator_name)
                approx_values = approximator.approximate(budget=budget, game=game)
                for order in range(1, exact_values.max_order + 1):
                    exact_values_order = exact_values.get_n_order(order)
                    approx_values_order = approx_values.get_n_order(order)
                    metrics = get_all_metrics(exact_values_order, approx_values_order)
                    run_result = {
                        "game": game.game_name,
                        "game_id": game.game_id,
                        "n_players": game.n_players,
                        "budget": budget,
                        "approximator": approximator.__class__.__name__,
                        "order": order,
                    }
                    run_result.update(metrics)
                    results.append(run_result)
                pbar.update(budget)

    results_df = pd.DataFrame(results)
    results_df.to_csv("results_all.csv", index=False)
