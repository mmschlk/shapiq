"""
Vision-Language Explainer — Custom Masks and Blur Occlusion
=============================================================

This example shows how to use the :class:`~shapiq.explainer.vision.VisionLanguageExplainer`
with a **custom segmenter** (user-provided binary masks) and a **Gaussian blur masker**,
instead of the default patch segmenter and mean occlusion.

The image is divided into three horizontal strips — top, middle, bottom — each treated
as a separate player. The ``CrossModalBlurMasker`` blurs occluded regions instead of
zeroing them out, which can produce more natural-looking occlusions for CNN-based models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from shapiq.explainer.vision import VisionLanguageExplainer
from shapiq.imputer.vision import MaskerConfig, SegmenterConfig

# %%
# 1. Load Model & Processor
# --------------------------

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
# 2. Load Input Image & Text
# ----------------------------
# Uses the bundled sample image. Replace with your own.

INPUT_TEXT = "black dog"

image_path = Path("tests") / "shapiq" / "data" / "dog_and_hydrant.png"
if not image_path.exists():
    image_path = Path("../../tests/shapiq/data/dog_and_hydrant.png")

image = Image.open(image_path).convert("RGB")
print(f"Image size: {image.size}")

# %%
# 3. Create Custom Player Masks
# -------------------------------
# We create 3 horizontal strips. Each strip is one player.
# The segmenter receives these as a (3, H, W) bool array.

W, H = image.size
masks = np.zeros((3, H, W), dtype=bool)
masks[0, : H // 3, :] = True  # top third
masks[1, H // 3 : 2 * H // 3, :] = True  # middle third
masks[2, 2 * H // 3 :, :] = True  # bottom third

seg_cfg = SegmenterConfig(strategy="custom_segmenter")

# %%
# 4. Configure a Blur Masker
# ----------------------------
# Instead of zeroing out pixels, ``crossmodal_blur`` applies a Gaussian blur
# to occluded regions. Requires ``scikit-image``.

msk_cfg = MaskerConfig(strategy="crossmodal_blur")

# %%
# 5. Build the Explainer
# ------------------------
# The :class:`~shapiq.explainer.vision.VisionLanguageExplainer` wraps the
# full pipeline. Image and text are passed at **explain time**.

explainer = VisionLanguageExplainer(
    model=model,
    processor=processor,
    segmenter_config=seg_cfg,
    masker_config=msk_cfg,
    batch_size=64,
    index="k-SII",
    max_order=2,
)

# %%
# 6. Compute Interaction Values
# -------------------------------
# The ``x`` parameter is a dict with ``"image"`` (PIL/ndarray/Tensor) and
# ``"text"`` (str). The explainer builds the imputer pipeline and approximator
# for this specific input.

iv = explainer.explain(
    x={"image": image, "text": INPUT_TEXT},
    budget=2**8,  # small budget for 5 players (3 image + 2 text from "black dog")
    custom_masks=masks,
)

print(iv)

# %%
# 7. Visualise Attributions
# ---------------------------
# The 3 image players correspond to the horizontal strips.
# Text players correspond to the tokenized input.

feature_names = ["Top", "Middle", "Bottom", "Token_1", "Token_2"]

# First-order (Shapley values)
fo = iv.get_n_order(1)
fo.plot_force(feature_names=feature_names, show=False)
plt.savefig("vision_explainer_force.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_explainer_force.png")

# Second-order interactions
iv.plot_network(feature_names=feature_names, show=False)
plt.savefig("vision_explainer_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_explainer_network.png")

# %%
# 8. Show the Custom Mask Layout
# --------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.imshow(np.array(image))
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[masks[i]] = [0.0, 0.8, 0.0, 0.4]  # green transparent
    ax.imshow(overlay)
    ax.set_title(f"Player {i}: {['Top', 'Middle', 'Bottom'][i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("vision_explainer_custom_masks.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_explainer_custom_masks.png")
