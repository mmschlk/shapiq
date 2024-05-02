"""This module contains the legacy test for the old language model game."""

# TODO: Remove this file after the benchmark is completed.

import numpy as np
import pandas as pd

from shapiq.games import Game
from shapiq.utils import powerset


class OldLMGame(Game):

    def __init__(self) -> None:
        super().__init__(n_players=14)
        data = pd.read_csv("old_lm_game_14_player.csv")
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
    df = pd.read_csv("old_lm_game_14_player.csv")
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
