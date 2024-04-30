"""This module contains the benchmark conducted on the language model game."""

import os
import sys
import glob

import numpy as np
from tqdm.auto import tqdm

import pandas as pd


from metrics import get_all_metrics
from precompute_lm import pre_compute_imdb

from shapiq.utils.sets import powerset
from shapiq.games import Game
from shapiq.approximator._base import Approximator


class OldLMGame(Game):

    def __init__(self) -> None:
        super().__init__(n_players=14)
        data = pd.read_csv("data/old_lm_game_14_player.csv")
        interaction_lookup, values = {}, []
        for i, row in data.iterrows():
            value = float(row["value"])
            interaction_id = row["set"]
            interaction = tuple()
            if interaction_id != "s_":
                interaction_id = interaction_id.split("s_")[1]
                interaction_id = [int(i) for i in interaction_id.split("_")]
                interaction = tuple(sorted(interaction_id))
            interaction_lookup[interaction] = len(values)
            values.append(value)
        self.coalition_lookup = interaction_lookup
        self.value_storage = np.array(values)
        self.precompute_flag = True
        self.normalization_value = float(self.value_storage[self.coalition_lookup[tuple()]])


def check_interaction(game_to_check: Game, verbose: bool = False) -> bool:
    """Checks if the values of the csv are the same as in the game.

    Args:
        game_to_check: The game to check the values for.
        verbose: Whether to print the values for each interaction.

    Returns:
        True if the values are the same, False otherwise.
    """
    check = True
    df = pd.read_csv("data/old_lm_game_14_player.csv")
    for i, interaction in enumerate(
        powerset(range(game_to_check.n_players), min_size=0, max_size=game_to_check.n_players)
    ):
        coalition = np.zeros(game_to_check.n_players, dtype=bool)
        coalition[list(interaction)] = True
        value_game = float(game_to_check(coalition))
        value_df = float(df.iloc[i]["value"])
        if value_game != value_df - game_to_check.normalization_value:
            print(
                f"Interaction {interaction} does not match with value {value_game} and "
                f"{value_df} - {game_to_check.normalization_value} = "
                f"{value_df - game_to_check.normalization_value}."
            )
            check = False
        if verbose:
            print(
                f"Row {i}: Interaction {interaction} does not match with value {value_game} and "
                f"{value_df} - {game_to_check.normalization_value} = "
                f"{value_df - game_to_check.normalization_value}."
            )
    return check


def get_approximator(approx_name: str) -> Approximator:
    """Returns an initialized approximator based on the name."""
    from shapiq.approximator import KernelSHAPIQ
    from shapiq.approximator import InconsistentKernelSHAPIQ
    from shapiq.approximator import SHAPIQ
    from shapiq.approximator import SVARMIQ
    from shapiq.approximator import PermutationSamplingSII

    if approx_name == "KernelSHAPIQ":
        return KernelSHAPIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    if approx_name == "InconsistentKernelSHAPIQ":
        return InconsistentKernelSHAPIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    if approx_name == "SHAPIQ":
        return SHAPIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    if approx_name == "SVARMIQ":
        return SVARMIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    if approx_name == "PermutationSamplingSII":
        return PermutationSamplingSII(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    raise ValueError(f"Invalid approximator name {approx_name}.")


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
    else:
        # load the pre-computed games
        files = glob.glob(
            os.path.join("precomputed", "SentimentAnalysis(Game)", str(N_PLAYERS), "*")
        )
        games = [Game(path_to_values=file) for file in files]

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
                approximator = get_approximator(approximator_name)
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
    results_df.to_csv("results.csv", index=False)
