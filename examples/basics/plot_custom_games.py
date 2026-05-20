"""
Defining Custom Games
=====================

This example demonstrates how to define custom cooperative games and use them
with :mod:`shapiq`. We model a simple cooking game and show how to query the
value function, precompute all coalition values, and save/load them.
"""

from __future__ import annotations

import numpy as np

import shapiq

# %%
# Introduction to Cooperative Game Theory
# ----------------------------------------
# Cooperative game theory deals with games in which players can form groups
# (coalitions) to achieve a collective payoff. A cooperative game is defined
# as a tuple :math:`(N, \\nu)` where :math:`N` is a finite set of players and
# :math:`\\nu: 2^N \\to \\mathbb{R}` maps every coalition to a real value.
#
# Defining a Custom Cooperative Game
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We consider a *cooking game* with three cooks: Alice, Bob, and Charlie.
#
# =============================  =======
# Coalition                      Quality
# =============================  =======
# {no cook}                      0
# {Alice}                        4
# {Bob}                          3
# {Charlie}                      2
# {Alice, Bob}                   9
# {Alice, Charlie}               8
# {Bob, Charlie}                 7
# {Alice, Bob, Charlie}          15
# =============================  =======


class CookingGame(shapiq.Game):
    """A cooperative game representing the cooking game with three cooks."""

    def __init__(self) -> None:
        self.characteristic_function = {
            (): 0,
            (0,): 4,
            (1,): 3,
            (2,): 2,
            (0, 1): 9,
            (0, 2): 8,
            (1, 2): 7,
            (0, 1, 2): 15,
        }
        super().__init__(
            n_players=3,
            player_names=["Alice", "Bob", "Charlie"],
            normalization_value=self.characteristic_function[()],
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        output = [self.characteristic_function[tuple(np.where(c)[0])] for c in coalitions]
        return np.array(output)


cooking_game = CookingGame()
print(cooking_game)

# %%
# Querying the Value Function
# ---------------------------
# We can query the value function with binary coalition vectors or player names.

coals = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
print("Values:", cooking_game(coals))

coals_named = [(), ("Alice", "Bob"), ("Alice", "Charlie"), ("Bob", "Charlie")]
print("Named:", cooking_game(coals_named))

print("Grand coalition value:", cooking_game.grand_coalition_value)
print("Empty coalition value:", cooking_game.empty_coalition_value)

# %%
# Precomputing Game Values
# ------------------------
# For small games we can precompute all :math:`2^n` coalition values.

cooking_game.precompute()
print("Precomputed values:", cooking_game.game_values)

# %%
# Saving and Loading
# ------------------
# Precomputed values can be saved and loaded from disk.

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "cooking_game_values.npz"
    cooking_game.save_values(save_path)

    loaded_game = shapiq.Game(n_players=3, path_to_values=save_path)
    print(loaded_game)
    print("Stored game values:", loaded_game.game_values)
