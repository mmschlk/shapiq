"""This module compares the new benchmark games lookup speed with the old benchmark games lookup
speed."""

import os
import random
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from shapiq.utils.sets import powerset, transform_coalitions_to_array
from shapiq.games import Game


class OldLookUpGame:
    """Wrapper for the Machine Learning Game to use precomputed model outputs for faster runtime in
    experimental settings."""

    def __init__(
        self,
        data_folder: str,
        n: int,
        data_id: Union[int, str] = None,
        used_ids: set = None,
        set_zero: bool = True,
        log_output: bool = False,
        min_max_normalize: bool = False,
        random_seed: int = None,
    ):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.n = n
        self.log_output = log_output

        # to not use the same instance twice if we use the game multiple times
        if used_ids is None:
            used_ids = set()
        self.used_ids = used_ids

        # get paths to the files containing the value function calls
        game_path = Path(__file__).parent.absolute()
        data_dir = os.path.join(game_path, "data")
        instances_dir = os.path.join(data_dir, data_folder, str(n))

        # randomly select a file if none was explicitly provided
        if data_id is None:
            files = os.listdir(instances_dir)
            files = list(set(files) - used_ids)
            files = sorted([file for file in files if file.endswith(".csv")])
            if len(files) == 0:
                files = os.listdir(instances_dir)
                self.used_ids = set()
            # select random file with seed
            data_id = random.choice(files)
            data_id = data_id.split(".")[0]
        self.data_id = str(data_id)
        self.game_name = "_".join(("LookUpGame", data_folder, str(n), self.data_id))

        self.used_ids.add(self.data_id + ".csv")

        # load file containing value functions into easily accessible dict
        file_path = os.path.join(instances_dir, self.data_id + ".csv")
        self.df = pd.read_csv(file_path)
        self.storage = {}
        for _, sample in self.df.iterrows():
            S_id = sample["set"]
            value = float(sample["value"])
            self.storage[S_id] = value

        if min_max_normalize:
            # normalize values to [0,1]
            storage_df = pd.DataFrame.from_dict(self.storage, orient="index")
            storage_df[0] = (storage_df[0] - storage_df[0].min()) / (
                storage_df[0].max() - storage_df[0].min()
            )
            self.storage = storage_df.to_dict()[0]

        # normalize empty coalition to zero (v_0({}) = 0)
        self.empty_value = 0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S):
        S_id = "s_" + "_".join([str(player) for player in sorted(S)])
        output = self.storage[S_id] - self.empty_value
        if self.log_output:
            return float(np.log(output))
        return output

    def get_name(self):
        return self.game_name


def time_both_games(n_sets: int) -> tuple[float, float]:
    """Time the new and old game for a given number of sets.

    Args:
        n_sets: The number of sets to test the games with.
    """
    n_players = 14

    # new game
    path_to_values = "precomputed/OldSentimentAnalysis(Game)/14/game.npz"
    new_game = Game(path_to_values=path_to_values)

    # old game
    old_game = OldLookUpGame(data_folder="OldSentimentAnalysis(Game)", n=n_players, set_zero=True)

    # new test coalitions
    test_coalitions_new = list(powerset(range(n_players), min_size=0, max_size=n_players))
    test_coalitions_new = transform_coalitions_to_array(test_coalitions_new, n_players=n_players)
    test_coalitions_new = np.tile(test_coalitions_new, (n_sets, 1))  # make coalitions 10x larger

    # old test coalitions
    test_coalitions_old = [
        set(coalition)
        for coalition in powerset(range(n_players), min_size=0, max_size=n_players)
        for _ in range(n_sets)
    ]

    # check that both have the same number of coalitions
    assert len(test_coalitions_new) == n_sets * 2**n_players
    assert len(test_coalitions_old) == n_sets * 2**n_players

    # time the new game
    start = time.time()
    _ = new_game(test_coalitions_new)
    new_time = time.time() - start

    # time the old game
    start = time.time()
    for coalition in test_coalitions_old:
        _ = old_game.set_call(coalition)
    old_time = time.time() - start

    return new_time, old_time


if __name__ == "__main__":

    N_SETS = [1, 2, 5, 10, 20]
    N_ITERATIONS = 10

    runtimes = []
    for sets in N_SETS:
        for _ in range(N_ITERATIONS):
            new, old = time_both_games(sets)
            runtimes.append({"sets": sets, "new": new, "old": old})

    runtimes_df = pd.DataFrame(runtimes)

    fig, ax = plt.subplots()
    for sets in N_SETS:
        runtimes_df_sets = runtimes_df[runtimes_df["sets"] == sets]
        new_time_mean = runtimes_df_sets["new"].mean()
        new_time_std = runtimes_df_sets["new"].std()
        old_time_mean = runtimes_df_sets["old"].mean()
        old_time_std = runtimes_df_sets["old"].std()

        ax.errorbar(int(sets), new_time_mean, yerr=new_time_std, fmt="o", c="orange")
        ax.errorbar(int(sets), old_time_mean, yerr=old_time_std, fmt="o", c="blue")

        print(f"New game took {new_time_mean:.6f} ± {new_time_std:.6f} seconds.")
        print(f"Old game took {old_time_mean:.6f} ± {old_time_std:.6f} seconds.")
        print(f"New game was {old_time_mean / new_time_mean:.2f} times faster than the old game.")

    ax.errorbar(0, 0, 0, fmt="o", c="orange", label="New game")
    ax.errorbar(0, 0, 0, fmt="o", c="blue", label="Old game")

    plt.xlabel("Number of sets")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()
