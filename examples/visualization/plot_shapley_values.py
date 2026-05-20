"""Computing Shapley Values.
=========================

A minimal example showing how to compute exact Shapley values for a small
cooperative game using :class:`~shapiq.ExactComputer`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import shapiq

# %%
# Define a Simple Game
# --------------------
# We define a 4-player weighted voting game as a callable.
# The game value is 1 if the coalition's total weight exceeds a threshold, else 0.

weights = np.array([0.4, 0.3, 0.2, 0.1])
threshold = 0.5


def voting_game(coalitions: np.ndarray) -> np.ndarray:
    """Return 1 if coalition weight exceeds the threshold, else 0."""
    return (coalitions @ weights > threshold).astype(float)


# %%
# Compute Exact Shapley Values
# ----------------------------
# :class:`~shapiq.ExactComputer` exhaustively evaluates all 2^n coalitions to
# compute exact interaction values. It accepts any callable game directly.

computer = shapiq.ExactComputer(voting_game, n_players=4)
sv = computer(index="SV", order=1)

# %%
# Visualize with a Bar Plot
# -------------------------

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(range(4), [sv[(i,)] for i in range(4)], color="#5b9bd5")
ax.set_xticks(range(4))
ax.set_xticklabels([f"Player {i}" for i in range(4)])
ax.set_ylabel("Shapley Value")
ax.set_title("Shapley Values for a Weighted Voting Game")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()
