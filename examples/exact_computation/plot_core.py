"""
The Core: A Different View on Explanation
=========================================

This example introduces the game-theoretic concept of the *core* and the
*egalitarian least core* (ELC), and shows how to compute them with
:class:`~shapiq.ExactComputer`.
"""

from __future__ import annotations

import numpy as np

import shapiq

# %%
# The Paper Game
# --------------
# Three AI researchers (Alice, Bob, Charlie) co-author a paper that wins a
# $500 best-paper award. We model their joint productivity as a cooperative
# game:
#
# =========================  =====
# Coalition                  Award
# =========================  =====
# {}                         $0
# {Alice}                    $0
# {Bob}                      $0
# {Charlie}                  $0
# {Alice, Bob}               $500
# {Alice, Charlie}           $400
# {Bob, Charlie}             $350
# {Alice, Bob, Charlie}      $500
# =========================  =====
#
# Shapley values give: Alice=$200, Bob=$175, Charlie=$125. But Alice and Bob
# could earn $250 each by excluding Charlie -- so the **core is empty**.
#
# The Least Core and ELC
# ~~~~~~~~~~~~~~~~~~~~~~
# The *least core* finds the minimum subsidy :math:`e` that makes cooperation
# stable. The *egalitarian least core* picks the distribution with minimum
# :math:`\|x\|_2` from the least core.


class PaperGame(shapiq.Game):
    def __init__(self) -> None:
        super().__init__(n_players=3, normalize=True, normalization_value=0)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        values = {
            (): 0,
            (0,): 0,
            (1,): 0,
            (2,): 0,
            (0, 1): 500,
            (0, 2): 400,
            (1, 2): 350,
            (0, 1, 2): 500,
        }
        return np.array([values[tuple(np.where(x)[0])] for x in coalitions])


paper_game = PaperGame()

# %%
# Computing the Egalitarian Least Core
# -------------------------------------

exact_computer = shapiq.ExactComputer(n_players=3, game=paper_game)
elc = exact_computer("ELC")
print("ELC payoffs:", elc.dict_values)
print("Stability subsidy e*:", exact_computer._elc_stability_subsidy)

# %%
# The ELC distributes approximately: Alice $233, Bob $183, Charlie $83,
# with a stability subsidy of about $83.
