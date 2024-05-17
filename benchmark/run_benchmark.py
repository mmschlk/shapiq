"""This module contains a setup and run configuration to run benchmarks on different games."""

import os
import glob

import numpy as np

from shapiq.games.benchmark import SHAPIQ_DATA_DIR, run_benchmark

from shapiq.approximator import (
    KernelSHAPIQ,
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    PermutationSamplingSII,
)

from shapiq.games import Game

if __name__ == "__main__":

    N_PLAYERS = 8  # number of players to load from
    N_GAMES = 5  # leave on 1 for debugging
    MAX_ORDER = 2
    INDEX = "SII"  # only SII or k-SII
    # games:
    # "SentimentAnalysis_Game",
    # "CaliforniaHousing_GlobalExplanation_Game",
    # "CaliforniaHousing_LocalExplanation_Game"
    GAME_NAME = "CaliforniaHousing_GlobalExplanation_Game"

    # define the budget steps
    MAX_BUDGET = min(2**N_PLAYERS, 10_000)
    BUDGET_STEPS = [int(budget_step * MAX_BUDGET) for budget_step in np.arange(0.15, 1.05, 0.05)]
    print("Budget steps: ", BUDGET_STEPS)

    # load the pre-computed games and get the exact values (gt_values)
    save_dir = os.path.join(SHAPIQ_DATA_DIR, GAME_NAME, str(N_PLAYERS))
    files = glob.glob(os.path.join(save_dir, "*.npz"))
    games = [Game(path_to_values=file) for i, file in enumerate(files) if i < N_GAMES]
    gt_values = [game.exact_values(index=INDEX, order=MAX_ORDER) for game in games]
    print("Loaded", len(games), "games.")

    # get approximators
    approximators = [
        KernelSHAPIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER),
        InconsistentKernelSHAPIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER),
        SHAPIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER),
        SVARMIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER),
        PermutationSamplingSII(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER),
    ]

    # run the benchmark
    results = run_benchmark(
        approximators=approximators,
        games=games,
        gt_values=gt_values,
        budget_steps=BUDGET_STEPS,
        n_iterations=1,
        n_jobs=6,
        save_path="results.csv",
    )
