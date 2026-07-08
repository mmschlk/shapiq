"""
Comparing Segmenters and Maskers for Vision-Language Explanations
==================================================================

Side-by-side comparison of the built-in segmenters and maskers on the same
image and text.

- ``patch``: rigid ViT-aligned grid.
- ``slic``: perceptual superpixels that follow content boundaries
  (required for CNN backbones).
- ``crossmodal_mean``: zero-out occlusion (dataset mean color).
- ``crossmodal_blur``: Gaussian-blur occlusion (softer, low-frequency
  information survives).

**Model:** ``openai/clip-vit-base-patch32``, text prompt ``"black dog"``.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from transformers import CLIPModel, CLIPProcessor

from shapiq.explainer.vision import VisionLanguageExplainer
from shapiq.imputer.vision import (
    MaskerConfig,
    SegmenterConfig,
    SlicParams,
    VisionImputer,
    VisionImputerFactory,
)

# 1. Load Model, Image & Text

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

INPUT_TEXT = "black dog"

image_path = Path("tests") / "shapiq" / "data" / "dog_and_hydrant.png"
if not image_path.exists():
    image_path = Path("../../tests/shapiq/data/dog_and_hydrant.png")
image = Image.open(image_path).convert("RGB")

# Helpers


def build_imputer(segmenter_strategy: str, masker_strategy: str) -> VisionImputer:
    """Build a VisionImputer for a segmenter/masker combination."""
    seg_cfg = make_segmenter_config(segmenter_strategy)
    msk_cfg = MaskerConfig(strategy=masker_strategy)
    return VisionImputerFactory().build(
        model, processor, image, INPUT_TEXT, segmenter_config=seg_cfg, masker_config=msk_cfg
    )


def make_segmenter_config(strategy: str) -> SegmenterConfig:
    """Fresh SegmenterConfig (the factory enriches configs in place)."""
    if strategy == "slic":
        return SegmenterConfig(strategy="slic", params=SlicParams(n_segments=49))
    return SegmenterConfig(strategy=strategy)


def player_label_map(imputer) -> np.ndarray:
    """Return a (H, W) int map: pixel -> image-player index."""
    identity = np.eye(imputer.n_players_image, dtype=bool)
    masks = imputer.segmenter.generate_masks(coalitions_image=identity)
    return masks.image_binary_mask[:, 0].numpy().argmax(axis=0)


def denormalize(pixel_values: np.ndarray) -> np.ndarray:
    """Undo CLIP normalization: (C, H, W) -> displayable (H, W, 3) in [0, 1]."""
    mean = np.array(processor.image_processor.image_mean).reshape(3, 1, 1)
    std = np.array(processor.image_processor.image_std).reshape(3, 1, 1)
    return np.clip(pixel_values * std + mean, 0, 1).transpose(1, 2, 0)


# 2. Segmenters

imputer_patch = build_imputer("patch", "crossmodal_mean")
imputer_slic = build_imputer("slic", "crossmodal_mean")

model_input = denormalize(imputer_patch.inputs_original.pixel_values[0].numpy())

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, imputer, title in (
    (axes[0], imputer_patch, "PatchSegmenter"),
    (axes[1], imputer_slic, "SLICSegmenter"),
):
    labels = player_label_map(imputer)
    ax.imshow(mark_boundaries(model_input, labels, color=(1, 1, 0), mode="thick"))
    ax.set_title(f"{title} — {imputer.n_players_image} image players")
    ax.axis("off")
plt.tight_layout()
plt.savefig("vision_comparison_segmenters.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_comparison_segmenters.png")

# 3. Maskers:
# Same coalitions for both maskers: all players kept, a random half occluded,
# and all occluded.

rng = np.random.default_rng(42)
n_players = imputer_patch.n_players_image
coalitions = np.stack(
    [
        np.ones(n_players, dtype=bool),
        rng.random(n_players) < 0.5,
        np.zeros(n_players, dtype=bool),
    ]
)
physical_mask = imputer_patch.segmenter.generate_masks(coalitions_image=coalitions)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
col_titles = ["full coalition (original)", "half occluded", "empty coalition (baseline)"]
for row, masker_strategy in enumerate(["crossmodal_mean", "crossmodal_blur"]):
    masker = build_imputer("patch", masker_strategy).masker
    masked = masker.apply(imputer_patch.inputs_original, physical_mask)
    for col in range(3):
        axes[row, col].imshow(denormalize(masked.pixel_values[col].numpy()))
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(col_titles[col])
    axes[row, 0].text(
        -0.08,
        0.5,
        masker_strategy,
        transform=axes[row, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )
plt.tight_layout()
plt.savefig("vision_comparison_maskers.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_comparison_maskers.png")

# 4. Attributions:
# Same image, text, and budget — only the segmenter/masker combination changes.

combos = [
    ("patch", "crossmodal_mean"),
    ("patch", "crossmodal_blur"),
    ("slic", "crossmodal_mean"),
    ("slic", "crossmodal_blur"),
]

results = []
for seg_strategy, msk_strategy in combos:
    explainer = VisionLanguageExplainer(
        model=model,
        processor=processor,
        segmenter_config=make_segmenter_config(seg_strategy),
        masker_config=MaskerConfig(strategy=msk_strategy),
        batch_size=64,
        index="SV",
        max_order=1,
        random_state=42,
    )
    start = time.perf_counter()
    sv = explainer.explain(x={"image": image, "text": INPUT_TEXT}, budget=2**7)
    runtime = time.perf_counter() - start
    imputer = explainer.game._imputer
    labels = player_label_map(imputer)
    values = np.array([sv[(i,)] for i in range(imputer.n_players_image)])
    results.append((seg_strategy, msk_strategy, values[labels], runtime))

max_abs = max(np.abs(heatmap).max() for *_, heatmap, _ in results)

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, (seg_strategy, msk_strategy, heatmap, _runtime) in zip(axes.flat, results, strict=False):
    ax.imshow(model_input)
    im = ax.imshow(heatmap, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, alpha=0.6)
    ax.set_title(f"{seg_strategy} + {msk_strategy}")
    ax.axis("off")
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03, label="Shapley value")
plt.savefig("vision_comparison_attributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_comparison_attributions.png")
