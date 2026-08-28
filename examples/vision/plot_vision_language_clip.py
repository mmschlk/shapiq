"""
Vision-Language Model Explanation with shapiq
===============================================

This example shows how to explain a CLIP model's similarity between an image and a text
description using the :class:`~shapiq.explainer.vision.VisionLanguageExplainer`.

*Requires only CPU, typical runtime < 1 minute.*

**Pipeline overview:**

1. **Segmenter** divides the input into "players" (image patches, text tokens).
2. **Masker** applies occlusion to a subset of those players.
3. **VisionImputer** orchestrates Segmenter → Masker → Model forward pass.
4. **VisionLanguageGame** adapts the imputer as a :class:`~shapiq.Game` for approximators.
5. Any **shapiq approximator** (e.g. :class:`~shapiq.KernelSHAPIQ`) computes Shapley interaction values.

**Model:** ``openai/clip-vit-base-patch32`` — 7x7 = 49 image patches, 2 text players.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import shapiq  # noqa: TC001
from shapiq.explainer.vision import VisionLanguageExplainer

# %%
# 1. Load Model & Processor
# --------------------------

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f"Image size: {model.config.vision_config.image_size}")
print(f"Patch size: {model.vision_model.embeddings.patch_size}")

# %%
# 2. Load Input Image & Text
# ----------------------------
# Uses the bundled sample image ``tests/shapiq/data/dog_and_hydrant.png``.
# Replace with your own image path for other experiments.

INPUT_TEXT = "black dog"

image_path = Path("tests") / "shapiq" / "data" / "dog_and_hydrant.png"
if not image_path.exists():
    image_path = Path("../../tests/shapiq/data/dog_and_hydrant.png")

image = Image.open(image_path).convert("RGB")
print(f"Image size: {image.size}")

# %%
# 3. Build the Explainer (default PatchSegmenter + CrossModalMeanMasker)
# -----------------------------------------------------------------------
# :class:`~shapiq.explainer.vision.VisionLanguageExplainer` wraps the
# full imputer pipeline. Image and text are passed at **explain time**.

explainer = VisionLanguageExplainer(
    model=model,
    processor=processor,
    batch_size=64,
    index="k-SII",
    max_order=2,
    random_state=42,
)

# Access the game and imputer built during explain()
iv = explainer.explain(
    x={"image": image, "text": INPUT_TEXT},
    budget=2**9,
)

game = explainer.game
imputer = explainer.game._imputer

print(f"Model type:           {imputer.model_type}")
print(f"Image grid:           {imputer.grid_size}x{imputer.grid_size}")
print(f"Image players:        {game.n_players_image}")
print(f"Text players:         {game.n_players_text}")
print(f"Total players:        {game.n_players}")
print(f"Empty coalition:      {game.empty_value:.4f}")
print(f"Full coalition:       {game.full_value:.4f}")
print(iv)

# %%
# 4a. Patch Grid — Visualise Player Layout
# ------------------------------------------
# Show the 7x7 patch grid with player indices and the tokenized text.

grid_size = imputer.grid_size
patch_size = imputer.patch_size
img_resized = image.resize((grid_size * patch_size, grid_size * patch_size))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.imshow(img_resized)
# Draw grid + label each patch with its player index
for row in range(grid_size):
    for col in range(grid_size):
        idx = row * grid_size + col
        cx = col * patch_size + patch_size // 2
        cy = row * patch_size + patch_size // 2
        rect = plt.Rectangle(
            (col * patch_size, row * patch_size),
            patch_size,
            patch_size,
            fill=False,
            edgecolor="white",
            lw=0.8,
            alpha=0.6,
        )
        ax1.add_patch(rect)
        ax1.text(
            cx,
            cy,
            str(idx),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.15", "facecolor": "black", "alpha": 0.5},
        )
ax1.set_title(f"Patch Grid ({grid_size}x{grid_size} = {game.n_players_image} players)")
ax1.axis("off")

# -- Tokenized text display --
text_tokens_raw = imputer.inputs_raw["input_ids"]
# Decode token IDs back to readable tokens
tokenizer = processor.tokenizer
text_tokens_decoded = tokenizer.convert_ids_to_tokens(text_tokens_raw[0].tolist())
# Strip CLIP special tokens (BOS at [0], EOS at [-1]) and byte-Pair artifacts
text_tokens = [t.replace("</w>", "") for t in text_tokens_decoded[1:-1]]

ax2.axis("off")
token_str = "\n".join([f"  [{i:2d}]  {tok}" for i, tok in enumerate(text_tokens)])
ax2.text(
    0.05,
    0.5,
    f"Text tokens ({game.n_players_text} players):\n\n{token_str}",
    transform=ax2.transAxes,
    fontsize=13,
    verticalalignment="center",
    family="monospace",
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "alpha": 0.9},
)
ax2.set_title("Tokenized Text (CLIP BOS/EOS stripped)")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("vision_clip_patch_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_clip_patch_grid.png")

print(f'Input text: "{INPUT_TEXT}"')
print(f"Number of image players: {game.n_players_image}")
print(f"Number of text players:  {game.n_players_text}")
print(f"Total players:           {game.n_players}")

# %%
# 4b. Force Plot (First-Order)
# ------------------------------
# Shows how each patch/token pushes the similarity score away from the baseline.

feature_names = [f"P{i}" for i in range(game.n_players_image)] + text_tokens

iv_first_order = iv.get_n_order(1)
iv_first_order.plot_force(feature_names=feature_names, show=False)
plt.savefig("vision_clip_force.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_clip_force.png")

# %%
# 4c. Interaction Network (Second-Order)
# ----------------------------------------
# Edges show pairwise interactions between image patches and text tokens.
# Blue = positive synergy (regions amplify each other), red = negative
# (they diminish each other).

iv.plot_network(feature_names=feature_names, draw_threshold=0.0, show=False)
plt.savefig("vision_clip_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_clip_network.png")

# %%
# 4d. Attribution Heatmap Overlay on Image
# ------------------------------------------
# A custom visualisation maps first-order Shapley values back to the original
# image grid. Green patches drive similarity **up** (important for the
# prediction); red patches drive it **down**.


def plot_patch_overlay(
    image: Image.Image,
    sv: shapiq.InteractionValues,
    grid_size: int,
    patch_size: int,
    max_abs_val: float | None = None,
    alpha: float = 0.6,
    cmap: str = "RdYlGn",
) -> plt.Figure:
    """Overlay patch-level Shapley values on the input image."""
    img_resized = image.resize((grid_size * patch_size, grid_size * patch_size))
    values = np.array([sv[(i,)] for i in range(grid_size * grid_size)])
    grid = values.reshape(grid_size, grid_size)

    if max_abs_val is None:
        max_abs_val = max(abs(grid.min()), abs(grid.max()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(img_resized)
    ax1.set_title("Original Image")
    ax1.axis("off")
    ax2.imshow(img_resized)
    im = ax2.imshow(
        grid,
        extent=(0, img_resized.width, img_resized.height, 0),
        cmap=cmap,
        vmin=-max_abs_val,
        vmax=max_abs_val,
        alpha=alpha,
    )
    # Draw patch boundaries
    for i in range(grid_size + 1):
        ax2.axhline(i * patch_size, color="white", lw=0.5, alpha=0.3)
        ax2.axvline(i * patch_size, color="white", lw=0.5, alpha=0.3)

    ax2.set_title("Patch Attribution (Shapley Values)")
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="k-SII Contribution")
    plt.tight_layout()
    return fig


fig = plot_patch_overlay(image, iv_first_order, imputer.grid_size, imputer.patch_size)
plt.savefig("vision_clip_patch_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_clip_patch_overlay.png")
