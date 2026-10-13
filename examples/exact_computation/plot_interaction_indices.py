"""
A Tour of the Interaction Indices
=================================

``shapiq`` computes many interaction indices, and they all answer a slightly different question.
This example computes nine of them exactly with :class:`~shapiq.game_theory.ExactComputer` on
one small game whose structure we know, and compares how each index distributes the same
interactions: a pairwise synergy, a pairwise redundancy, and an interaction of four players that
no explanation of order two can show as it is.

Along the way, we check which indices are *efficient*, i.e. sum up to
:math:`\\nu(N) - \\nu(\\emptyset)`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import shapiq

# %%
# A Game with a Known Structure
# -----------------------------
# Eight players take part in the game :math:`\nu`:
#
# - **A** and **B** are *complements*: each is worth 1 alone, but 3 together. Their **synergy**
#   adds 1 when both are present.
# - **C** and **D** are *substitutes*: one of them is enough to earn 1. Their **redundancy**
#   subtracts 1 when both are present.
# - **E**, **F**, **G**, and **H** form a *team* that earns 2, but only if all four are present:
#   an **interaction of order four**.
#
# The value of the grand coalition is :math:`\nu(N) = 3 + 1 + 2 = 6`, and the empty coalition is
# worth :math:`\nu(\emptyset) = 0`.

PLAYERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
N_PLAYERS = len(PLAYERS)


def game(coalitions: np.ndarray) -> np.ndarray:
    """The value of every coalition (one row of a boolean matrix per coalition)."""
    a, b, c, d, e, f, g, h = np.atleast_2d(coalitions).astype(float).T
    synergy = a + b + a * b  # complements: 1 + 1 alone, 3 together
    redundancy = np.maximum(c, d)  # substitutes: one of them is enough
    team = 2 * e * f * g * h  # all four are needed
    return synergy + redundancy + team


def name(interaction: tuple[int, ...]) -> str:
    """Name an interaction by its players, e.g. ``(0, 1)`` as ``"AB"``."""
    return "".join(PLAYERS[player] for player in interaction) or "{}"


computer = shapiq.ExactComputer(game, n_players=N_PLAYERS)
total = computer.game_values[-1] - computer.baseline_value  # the grand coalition comes last
print(f"v(N) - v(empty) = {total}")

# %%
# Nine Indices of the Same Game
# -----------------------------
# We compute nine indices of the game:
#
# - the Möbius transform :cite:p:`Rota.1964`, which describes the game exactly with a value
#   for every coalition (see below),
# - the Shapley value (SV) :cite:p:`Shapley.1951` and the Banzhaf value (BV)
#   :cite:p:`Banzhaf.1964` attribute the game to the players,
# - the Shapley and Banzhaf interaction indices (SII and BII) :cite:p:`Grabisch.1999` measure
#   the interaction of a group of players in the context of all other players,
# - the k-Shapley values (k-SII) :cite:p:`Bordt.2023`, the Shapley-Taylor interaction index
#   (STII) :cite:p:`Sundararajan.2020`, and the faithful Shapley and Banzhaf interaction
#   indices (FSII and FBII) :cite:p:`Tsai.2022` are explanations up to an order :math:`k`:
#   they divide the game among all interactions of at most :math:`k` players. We use
#   :math:`k = 2`: values for single players and for pairs.
#
# :class:`~shapiq.game_theory.ExactComputer` evaluates the game once on all :math:`2^8 = 256`
# coalitions and computes every index from these values.

INDICES = {  # index: highest order
    "Moebius": N_PLAYERS,
    "SV": 1,
    "BV": 1,
    "SII": 2,
    "BII": 2,
    "k-SII": 2,
    "STII": 2,
    "FSII": 2,
    "FBII": 2,
}
results = {index: computer(index=index, order=order) for index, order in INDICES.items()}
print(results["k-SII"])

# %%
# How Each Index Distributes the Interactions
# --------------------------------------------
# The heatmap shows every value that is not zero for at least one index, grouped by the three
# parts of the game. An empty cell marks an interaction that the index does not describe (pairs
# for the values of order 1, the team of four for the indices of order 2).

columns = [(0,), (1,), (0, 1), (2,), (3,), (2, 3)]  # synergy and redundancy
columns += [(i,) for i in range(4, 8)]  # the members of the team
columns += [(i, j) for i in range(4, 8) for j in range(i + 1, 8)]  # the pairs of the team
columns += [(4, 5, 6, 7)]  # the team itself
table = np.full((len(results), len(columns)), np.nan)
for row, values in enumerate(results.values()):
    for col, interaction in enumerate(columns):
        if len(interaction) <= values.max_order:
            table[row, col] = values[interaction]

# shapiq's colors: red for positive, blue for negative values, gray in between
cmap = LinearSegmentedColormap.from_list("shapiq", ["#1e88e5", "#f0efec", "#ff0d57"])
cmap.set_bad("white")
limit = np.nanmax(np.abs(table))
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.imshow(np.ma.masked_invalid(table), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
for (row, col), value in np.ndenumerate(table):
    if not np.isnan(value):
        label = "0" if abs(value) < 1e-10 else f"{value:.2f}"
        ax.text(col, row, label, ha="center", va="center", fontsize=8.5)
ax.set_xticks(range(len(columns)), [name(interaction) for interaction in columns])
ax.set_yticks(range(len(results)), list(results))
ax.tick_params(length=0)
for boundary in (2.5, 5.5):  # between the three parts of the game
    ax.axvline(boundary, color="white", linewidth=4)
for position, title in ((1, "synergy"), (4, "redundancy"), (11, "team of four")):
    ax.text(position, -0.8, title, ha="center", va="bottom", fontsize=10)
ax.set_title("Exact interaction values of the same game", pad=24)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.show()

# %%
# The Möbius Transform: the Building Blocks of the Game
# -----------------------------------------------------
# The first row of the heatmap is the Möbius transform :cite:p:`Rota.1964`. Every game is a
# sum of *pure* interactions, one for every coalition :math:`T`, and the Möbius transform gives
# their weights:
#
# .. math::
#     m(T) = \sum_{L \subseteq T} (-1)^{|T| - |L|} \, \nu(L),
#     \qquad \nu(S) = \sum_{T \subseteq S} m(T).
#
# It is the only index here that describes the game exactly: it has a value for interactions of
# every order, up to :math:`2^n` of them. For our game, only seven are not zero, and they are
# exactly the structure we built in. The UpSet plot shows them.

moebius = results["Moebius"]
for interaction, value in moebius.dict_values.items():
    if abs(value) > 1e-12:
        print(f"m({name(interaction)}) = {value:g}")

moebius.plot_upset(feature_names=PLAYERS, n_interactions=7, show=True)

# %%
# Reading the Heatmap
# -------------------
# The other rows summarize these seven interactions in values of order 1 or 2.
#
# **Synergy and redundancy.** All indices of order 2 put the full Möbius value on the pairs AB
# (:math:`+1`) and CD (:math:`-1`): for interactions of at most :math:`k` players, they agree with
# the Möbius transform. They differ in the single players. SII (and SV) give A the value 1.5, its
# own effect plus half of the synergy; k-SII, STII, and FSII give A only its own effect of 1,
# because the synergy is already shown on the pair. The values of order 1, SV and BV, have no
# pairs and split the synergy (and the redundancy) equally among the two players.
#
# **The team of four.** An index of order 2 has no value for the team, so each index distributes
# its 2 differently among the four members and their six pairs:
#
# - SV gives every member :math:`2/4 = 0.5`; BV only :math:`0.25`, half of it is lost.
# - SII and BII measure the pairs in the context of the other players: :math:`2/3` and
#   :math:`1/2` per pair, on top of the full Shapley or Banzhaf value of each member.
# - STII moves the whole interaction to the highest order: :math:`2/6` on each of the six pairs,
#   nothing on the members.
# - k-SII and FSII put more than the whole team value on the pairs and correct it with negative
#   values for the members; they distribute it differently (:math:`2/3` and :math:`-1/2`
#   versus :math:`0.6` and :math:`-0.4`).
# - FBII, the faithful index of the Banzhaf value, gives :math:`0.5` to each pair and
#   :math:`-0.5` to each member.
#
# None of these is wrong: every index of order 2 is a different summary of an interaction that
# order 2 cannot show. FSII is the summary that approximates the game best in a weighted
# least-squares sense, STII the one that attributes higher-order effects only to the top order,
# and k-SII the one that turns the SII into an efficient explanation.

# %%
# Which Indices Are Efficient?
# ----------------------------
# An efficient index divides exactly the payout :math:`\nu(N) - \nu(\emptyset)` among its
# interactions. We sum all values of order 1 and higher (the value of the empty set is the
# baseline, not an attribution).

sums = {
    index: sum(value for interaction, value in values.dict_values.items() if len(interaction) > 0)
    for index, values in results.items()
}
efficient = {index: bool(np.isclose(value, total)) for index, value in sums.items()}
for index, value in sums.items():
    print(f"{index:>8}: sum = {value:6.3f}, efficient: {efficient[index]}")

fig, ax = plt.subplots(figsize=(7, 3.6))
positions = np.arange(len(sums))
colors = ["#2a78d6" if efficient[index] else "#eb6834" for index in sums]
ax.barh(positions, list(sums.values()), color=colors, height=0.6)
ax.axvline(total, color="#52514e", linestyle="--", linewidth=1.5)
ax.text(total, -0.6, r" $\nu(N) - \nu(\emptyset)$", va="bottom", color="#52514e")
for position, value in zip(positions, sums.values(), strict=True):
    ax.text(value + 0.1, position, f"{value:.2f}", va="center", fontsize=9)
ax.set_yticks(positions, list(sums))
ax.set_ylim(len(sums) - 0.5, -1.2)  # first index on top, room for the label
ax.set_xlabel("Sum of the values of order 1 and higher")
ax.set_xlim(0, max(sums.values()) * 1.12)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Efficient (blue) and not efficient (orange) indices")
plt.tight_layout()
plt.show()

# %%
# The Möbius transform, SV, k-SII, STII, and FSII are efficient. BV and FBII miss the payout
# (and FBII brings its own value for the empty set), and SII and BII exceed it: their pairs are
# measured in addition to the single players, not instead of them.
#
# In short:
#
# - Use **SV** (or **BV**) for one value per player.
# - Use **SII** (or **BII**) to measure how strongly a group of players interacts, not to divide
#   the payout.
# - Use **k-SII**, **STII**, or **FSII** for an efficient explanation with interactions of up to
#   :math:`k` players. They agree on interactions of at most :math:`k` players and differ in how
#   they show larger ones.
# - Use the **Möbius transform** for the complete, exact picture, if the number of players is
#   small or the game is sparse.
