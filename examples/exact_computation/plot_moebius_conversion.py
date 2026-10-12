"""
The Möbius Transform and Conversions Between Indices
====================================================

Every interaction index in ``shapiq`` is a linear function of the Möbius transform of the game.
If the Möbius transform is known, every index follows from it without evaluating the game again.
This example shows how to

- compute the Möbius transform of a game and rebuild the game from it,
- convert it into Shapley values and interactions with
  :class:`~shapiq.game_theory.MoebiusConverter`,
- compute exact interactions of a sparse game with 60 players, where brute force would need
  :math:`2^{60}` evaluations, and
- turn the Shapley interaction index (SII) into the efficient k-SII with
  :func:`~shapiq.game_theory.aggregate_base_interaction`.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

import shapiq
from shapiq.game_theory import MoebiusConverter, aggregate_base_interaction
from shapiq_games import SOUM

# %%
# From the Game to the Möbius Transform and Back
# ----------------------------------------------
# The Möbius transform :footcite:p:`Rota.1964` assigns a value :math:`m(T)` to every coalition
# :math:`T` such that the value of any coalition is the sum of the Möbius values of its subsets:
#
# .. math::
#     m(T) = \sum_{L \subseteq T} (-1)^{|T| - |L|} \, \nu(L),
#     \qquad \nu(S) = \sum_{T \subseteq S} m(T).
#
# The two formulas are the two directions of the same one-to-one map
# :footcite:p:`Grabisch.2000`: the Möbius transform is not a summary but another way to write
# down the game. We use a sum of unanimity games (:class:`~shapiq_games.SOUM`) with eight players:
# a game built from twelve random interactions, so its Möbius transform has (at most) twelve
# values that are not zero.

game = SOUM(n=8, n_basis_games=12, max_interaction_size=4, random_state=1)
computer = shapiq.ExactComputer(game)  # evaluates the game on all 2**8 coalitions
moebius = computer(index="Moebius")

nonzero = {t: v for t, v in moebius.dict_values.items() if abs(v) > 1e-12}
print(f"{len(nonzero)} of {2**game.n_players} Möbius values are not zero:")
for interaction, value in sorted(nonzero.items(), key=lambda item: (len(item[0]), item[0])):
    print(f"  m{interaction} = {value:+.4f}")

# %%
# The game knows its own Möbius transform (it is built from it), and the transform computed from
# the 256 game values agrees with it. Summing the Möbius values over the subsets of a coalition
# gives back its value, for all 256 coalitions.

known = game.moebius_coefficients
same = all(np.isclose(moebius[interaction], v) for interaction, v in known.dict_values.items())
print("Same as the known transform:", same)

coalitions = np.array(
    [[(i >> p) & 1 for p in range(game.n_players)] for i in range(2**8)], dtype=bool
)
rebuilt = np.array(
    [
        sum(value for interaction, value in nonzero.items() if coalition[list(interaction)].all())
        for coalition in coalitions
    ]
)
print("Game rebuilt from the Möbius transform:", np.allclose(rebuilt, game(coalitions)))

# %%
# Converting the Möbius Transform into Other Indices
# --------------------------------------------------
# Every index distributes each Möbius value :math:`m(T)` onto the interactions
# :math:`S \subseteq T` it describes, with a weight that only depends on the sizes :math:`|S|` and
# :math:`|T|` (and the order of the explanation). The Shapley value, for instance, splits :math:`m(T)` equally
# among the players of :math:`T`. :class:`~shapiq.game_theory.MoebiusConverter` applies these
# weights: it needs only the Möbius values that are not zero, not the game. We check that it
# agrees with :class:`~shapiq.game_theory.ExactComputer`, which computes each index from the 256
# game values.

converter = MoebiusConverter(moebius)
CONVERSIONS = [
    ("SV", 1),
    ("BV", 1),
    ("SII", 2),
    ("k-SII", 2),
    ("STII", 2),
    ("FSII", 2),
    ("FBII", 2),
]
for index, order in [*CONVERSIONS, ("k-SII", 3), ("FSII", 3)]:
    from_moebius = converter(index=index, order=order)
    exact = computer(index=index, order=order)
    interactions = {t for t in exact.dict_values if len(t) > 0}  # the empty set is the baseline
    difference = max(abs(from_moebius[t] - exact[t]) for t in interactions)
    print(f"{index:>5} up to order {order}: largest difference {difference:.1e}")

# %%
# The differences are rounding errors (FSII is a least-squares solution in
# :class:`~shapiq.game_theory.ExactComputer`, hence its larger, but still tiny, difference).
#
# How One Interaction Is Distributed
# ----------------------------------
# The distribution weights are easiest to see on a game with a single Möbius value: the
# unanimity game of a coalition :math:`T` of :math:`t` players, which is worth 1 if all of
# :math:`T` is present and 0 otherwise. We convert :math:`m(T) = 1` for teams of 1 to 8 players
# and read off the value that every index of order 2 assigns to each player of :math:`T` and to
# each pair of players in :math:`T`.

INDICES = ["SII", "k-SII", "STII", "FSII"]
team_sizes = np.arange(1, 9)
per_player = {index: [] for index in ["SV", *INDICES]}
per_pair = {index: [] for index in INDICES}
for size in team_sizes:
    team = tuple(range(size))
    single = shapiq.InteractionValues(
        values=np.array([1.0]),
        index="Moebius",
        max_order=8,
        min_order=0,
        n_players=8,
        interaction_lookup={team: 0},
        estimated=False,
        baseline_value=0.0,
    )
    converter = MoebiusConverter(single)
    per_player["SV"].append(converter(index="SV", order=1)[(0,)])
    for index in INDICES:
        values = converter(index=index, order=2)
        per_player[index].append(values[(0,)])
        per_pair[index].append(values[(0, 1)] if size > 1 else np.nan)

COLORS = {
    "SV": "#52514e",
    "SII": "#2a78d6",
    "k-SII": "#eb6834",
    "STII": "#1baf7a",
    "FSII": "#6250d6",
}
MARKERS = {"SV": "s", "SII": "o", "k-SII": "D", "STII": "^", "FSII": "v"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, values, title in (
    (axes[0], per_player, "Value of each player of the team"),
    (axes[1], per_pair, "Value of each pair of players of the team"),
):
    for index, series in values.items():
        style = {"marker": MARKERS[index], "color": COLORS[index], "label": index, "lw": 2}
        if index == "SII":  # wide and hollow: SII coincides with SV and with k-SII
            style |= {"lw": 6, "alpha": 0.45, "markersize": 12, "markerfacecolor": "none"}
        ax.plot(team_sizes, series, **style)
    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
    ax.set_title(title)
    ax.set_xlabel("Size of the team (the interaction of the game)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Value (the team is worth 1)")
axes[0].legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
# For teams of one or two players, all indices of order 2 agree: they show the interaction as it
# is. Larger teams are interactions that order 2 cannot show, and the indices summarize them
# differently:
#
# - The Shapley value gives :math:`1/t` to each player.
# - SII gives each player its Shapley value and each pair :math:`1/(t-1)`, so its values add up
#   to more than the team is worth.
# - STII puts the whole team value on the pairs (:math:`1/\binom{t}{2}` each) and nothing on the
#   players.
# - k-SII :footcite:p:`Bordt.2023` keeps the SII values of the pairs and corrects the players with
#   negative values, so that the values add up to the worth of the team.
# - FSII :footcite:p:`Tsai.2022` also gives negative values to the players, but smaller pair
#   values: it is the best approximation of the game by interactions of up to two players in a
#   weighted least-squares sense.
#
# Exact Interactions of a Large, Sparse Game
# -------------------------------------------
# The converter works with the Möbius values that are not zero, so its cost grows with their
# number, not with :math:`2^n`. A sum of 40 unanimity games with 60 players has at most 40
# Möbius values that are not zero, and the converter computes its exact interactions in
# milliseconds, while brute force would need :math:`2^{60} \approx 10^{18}` evaluations of the
# game. This is how :class:`~shapiq_benchmark.computers.MoebiusComputer` computes the ground truth
# of the synthetic games in benchmarks.

large_game = SOUM(
    n=60, n_basis_games=40, min_interaction_size=1, max_interaction_size=6, random_state=0
)
start = time.perf_counter()
large_converter = MoebiusConverter(large_game.moebius_coefficients)
sv = large_converter(index="SV", order=1)
k_sii = large_converter(index="k-SII", order=2)
print(f"SV and k-SII of a 60-player game in {time.perf_counter() - start:.3f} s")
payout = large_game.grand_coalition_value - large_game.empty_coalition_value
print("Shapley values are efficient:", np.isclose(sum(sv.get_n_order_values(1)), payout))
print(f"Non-zero k-SII values: {np.count_nonzero(k_sii.values)} of {1 + 60 + 60 * 59 // 2}")
print("The five largest k-SII values:")
for interaction, value in sorted(k_sii.dict_values.items(), key=lambda item: -abs(item[1]))[:5]:
    print(f"  {interaction}: {value:+.4f}")

# %%
# From SII to k-SII
# -----------------
# SII measures each interaction in isolation and is not efficient. k-SII :footcite:p:`Bordt.2023`
# turns the SII values up to order :math:`k` into an efficient explanation: the SII values of
# higher orders are projected onto the lower orders with weights given by the Bernoulli numbers.
# :func:`~shapiq.game_theory.aggregate_base_interaction` performs this aggregation on any
# :class:`~shapiq.interaction_values.InteractionValues` of the index SII (for example estimates of
# an approximator), and gives exactly the k-SII of the converter.

converter = MoebiusConverter(moebius)  # the eight-player game from above
sii = converter(index="SII", order=2)
aggregated = aggregate_base_interaction(sii)
k_sii_small = converter(index="k-SII", order=2)
difference = max(abs(aggregated[t] - k_sii_small[t]) for t in k_sii_small.dict_values if len(t) > 0)
print(f"Index after aggregation: {aggregated.index}, largest difference to k-SII: {difference:.1e}")

total = game.grand_coalition_value - game.empty_coalition_value
for name, values in (("SII", sii), ("k-SII", aggregated)):
    payout = sum(value for interaction, value in values.dict_values.items() if len(interaction) > 0)
    print(f"{name:>5}: sum = {payout:+.4f}, v(N) - v(empty) = {total:+.4f}")

# %%
# References
# ----------
# .. footbibliography::
