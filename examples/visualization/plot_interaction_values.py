"""
Working with InteractionValues
==============================

This example shows how to create, manipulate, aggregate, and visualize
:class:`~shapiq.InteractionValues` objects -- the central output type of
``shapiq``.
"""

from __future__ import annotations

from shapiq import InteractionValues

# %%
# Creating InteractionValues
# --------------------------
# Shapley values (first-order interactions) for 3 players:

sv = InteractionValues(
    values={(0,): 1.0, (1,): 1.0, (2,): 1.0},
    index="SV",
    max_order=1,
    n_players=3,
    min_order=1,
)
print(sv)

# %%
# Second-order Shapley Interaction Index (SII) values:

raw_scores = {
    (): 0,
    (0,): 1,
    (1,): 1,
    (2,): 1,
    (0, 1): 0.5,
    (0, 2): 0.5,
    (1, 2): 0.5,
}
sii = InteractionValues(
    values=raw_scores,
    index="SII",
    max_order=2,
    n_players=3,
    min_order=0,
)
print(sii)

# %%
# Manipulating InteractionValues
# ------------------------------
# Scalar operations and pairwise addition/subtraction are supported.

scaled = sv + 2
scaled *= 2
scaled -= 2

sii2 = InteractionValues(
    values=raw_scores,
    index="SII",
    max_order=2,
    n_players=3,
    min_order=0,
)
added = sii + sii2
subtracted = sii - sii2
aggregated = sii.aggregate([sii2], aggregation="mean")

# %%
# Extracting Specific Orders
# ---------------------------
# Get only second-order interactions, or as a matrix.

print(sii.get_n_order(2))
print("Second-order matrix:\n", sii.get_n_order_values(2))

# %%
# Subset of Players
# ------------------

print(sii.get_subset([0, 2]))

# %%
# Sparsification
# ---------------
# Remove near-zero interactions.

noisy = InteractionValues(
    values={(0,): 1.0, (1,): 1.0, (2,): 1.0, (0, 1): 1e-5, (0, 2): 1e-3, (1, 2): -1e-5},
    index="SII",
    max_order=2,
    n_players=3,
    min_order=0,
)
noisy.sparsify(threshold=1e-3)
print("After sparsify:", noisy.interactions)

# %%
# Visualization
# -------------
# Force plot and UpSet plot.

sii.plot_force()

# %%

sii.plot_upset()

# %%
# Saving and Loading
# ------------------

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "sii.json"
    sii.save(path)
    loaded = InteractionValues.load(path)
    print(loaded)
