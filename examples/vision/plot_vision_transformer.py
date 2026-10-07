"""
Explaining a Vision Transformer
=================================

In an image classification game, the players are regions of an image, and the value of a
coalition is the probability of a class when only those regions are visible. This example
explains a Vision Transformer (ViT) on a photo from Imagenette, a ten-class subset of ImageNet,
with the :class:`~shapiq_games.ImageClassifier` game: it shows what the players are and what the
model sees, computes exact Shapley values and interactions, and draws them on the image.
"""

from __future__ import annotations

import os

# Prevent OpenMP/MKL thread conflicts with PyTorch backend
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

import shapiq
from shapiq_games import ImageClassifier
from shapiq_games.datasets import load_imagenette

# %%
# The Game
# --------
# :func:`~shapiq_games.datasets.load_imagenette` downloads the Imagenette validation images
# once and caches them. Every image keeps its ImageNet class, so the pretrained ViT classifies it
# directly. We explain an English springer for its true class. With ``"vit_9_patches"``, the
# players are a 3x3 grid of the ViT's patches.

images = load_imagenette(split="val")
image, label = images[388], int(images.labels[388])
game = ImageClassifier(image, model="vit_9_patches", class_index=label)
print(f"{game.n_players} players, explained class: {game.class_name}")

# %%
# What the Players Are and What the Model Sees
# ---------------------------------------------
# :attr:`~shapiq_games.ImageClassifier.regions` assigns every pixel to a player. For a coalition,
# the ViT sees only the patches of its players; the others are removed inside the model with a
# mask token (shown in gray by :meth:`~shapiq_games.ImageClassifier.masked_image`).

coalition = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0], dtype=bool)
probability = game(coalition.reshape(1, -1))[0] + game.normalization_value

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(game.image)
axes[0].set_title("Image")
axes[1].imshow(game.regions, cmap="tab10")
axes[1].set_title("Players")
axes[2].imshow(game.masked_image(coalition))
axes[2].set_title(f"A coalition: p = {probability:.2f}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# Exact Values
# ------------
# With 9 players, :class:`~shapiq.ExactComputer` evaluates all 512 coalitions once; every index
# is then exact.

computer = shapiq.ExactComputer(game)
shapley_values = computer("SV", 1)
interactions = computer("k-SII", 2)

# %%
# Shapley Values on the Image
# ---------------------------
# :meth:`~shapiq_games.ImageClassifier.attribution_map` spreads each player's value over its
# pixels: red regions raise the probability of the class, blue regions lower it.

heatmap = game.attribution_map(shapley_values)
bound = np.abs(heatmap).max()
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(game.image)
overlay = ax.imshow(heatmap, cmap="RdBu_r", alpha=0.55, vmin=-bound, vmax=bound)
fig.colorbar(overlay, ax=ax, shrink=0.8, label="Shapley value")
ax.axis("off")
plt.show()

# %%
# Interactions Between Patches
# ----------------------------
# The explanation graph shows each patch as a node, with its image from
# :meth:`~shapiq_games.ImageClassifier.player_images`; edges show the pairwise interactions.

shapiq.si_graph_plot(
    interactions,
    feature_image_patches=game.player_images(),
    center_image=game.image,
    show=True,
)
