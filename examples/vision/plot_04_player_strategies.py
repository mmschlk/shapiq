"""
Defining Players for Image Explanations
=======================================

Shapley values are assigned per player, so the player strategy decides which
units your explanation talks about. ``shapiq.vision`` ships four player
strategies, selected with the ``player_strategy`` argument of the
architecture:

- :class:`~shapiq.vision.SuperpixelStrategy` -- automatic SLIC superpixels
  (the pixel-space default, follows image structure)
- :class:`~shapiq.vision.players.GridStrategy` -- a fixed rectangular grid
  (content-independent, comparable across images)
- :class:`~shapiq.vision.players.CustomPlayerStrategy` -- user-supplied
  masks, e.g. from an external segmentation algorithm or model
- :class:`~shapiq.vision.PatchStrategy` -- groups of patch tokens (the
  token-space default for Vision Transformers)

The first three are pixel-space strategies for
:class:`~shapiq.vision.ClassificationArchitecture`; ``PatchStrategy`` belongs to
:class:`~shapiq.vision.ViTClassificationArchitecture`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from huggingface_hub import logging as hf_logging
from PIL import Image
from torchvision import models, transforms

hf_logging.set_verbosity_error()  # hide hub warnings (e.g. unauthenticated requests)
transformers.logging.set_verbosity_error()  # hide download noise and load reports
transformers.utils.logging.disable_progress_bar()

# %%
# Image and CNN as in the quickstart -- the ImageNet normalization lives
# inside the model because ``shapiq`` feeds it images scaled to ``[0, 1]``
# (see :ref:`sphx_glr_auto_examples_vision_plot_01_quickstart_cnn.py`).

resize_and_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
pil_image = Image.open(Path("imagenet_sample.png")).convert("RGB")
image = np.array(resize_and_crop(pil_image))

model = torch.nn.Sequential(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
)
model = model.eval()

# %%
# The Pixel-Space Strategies
# --------------------------
# :class:`~shapiq.vision.SuperpixelStrategy` runs SLIC internally and may
# return slightly more segments than requested. ``GridStrategy`` accepts
# either ``grid_shape`` (fix the number of tiles) or ``patch_size`` (fix the
# tile size in pixels; ``patch_size=56`` on a 224x224 image is equivalent to
# ``grid_shape=4``). ``CustomPlayerStrategy`` accepts a boolean
# ``(n_players, H, W)`` array or a 2-D integer label map -- here the output
# of a fine-grained SLIC run stands in for any external segmenter.

from skimage.segmentation import mark_boundaries, slic

from shapiq.vision import SuperpixelStrategy
from shapiq.vision.players import CustomPlayerStrategy, GridStrategy

strategies = {
    "SuperpixelStrategy": SuperpixelStrategy(n_segments=12),
    "GridStrategy": GridStrategy(grid_shape=4),
    "CustomPlayerStrategy": CustomPlayerStrategy(slic(image, n_segments=64, start_label=0)),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, strategy) in zip(axes, strategies.items(), strict=True):
    masks = strategy.get_masks(image)
    ax.imshow(mark_boundaries(image, masks.argmax(axis=0)))
    ax.set_title(f"{name}: {strategy.n_players} players")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# Pixels not covered by any custom mask stay visible in every coalition and
# cannot be attributed; ``CustomPlayerStrategy`` raises a ``UserWarning``
# when it detects such gaps.

# %%
# Selecting a Pixel Strategy
# --------------------------
# Pass the strategy to :class:`~shapiq.vision.ClassificationArchitecture`. The same
# explanation three times -- same model, same image, same masking -- and
# only the unit of explanation changes.

from shapiq.vision import ClassificationArchitecture, ImageExplainer

for name, strategy in strategies.items():
    architecture = ClassificationArchitecture(model=model, player_strategy=strategy)
    explainer = ImageExplainer(model=architecture, data=image, random_state=42)
    interaction_values = explainer.explain(budget=256)
    fig, ax = interaction_values.plot_image_attributions(
        image, explainer.imputer.player_masks, show=False
    )
    ax.set_title(f"{name} ({explainer.imputer.n_features} players)")
    plt.show()

# %%
# Superpixels adapt to image content, so region boundaries tend to follow
# objects -- usually the best default for inspecting a single image. A grid
# is content-independent, which makes attributions comparable across images
# and is the natural choice for benchmarks. Custom masks are the right tool
# when you already know the semantic parts (foreground vs. background,
# objects, organs).

# %%
# Token Players for Vision Transformers: PatchStrategy
# ----------------------------------------------------
# ViTs explained with token masking (see
# :ref:`sphx_glr_auto_examples_vision_plot_02_quickstart_vit.py`) define
# players as groups of patch tokens with
# :class:`~shapiq.vision.PatchStrategy`. ``n_players`` must be a perfect
# square whose square root divides the model's token grid; the default is
# coarse (4 quadrant players on ViT-base's 14x14 grid). On a 14x14 grid, 49
# players gives each player a 2x2 block of tokens:

from transformers import AutoImageProcessor, AutoModelForImageClassification

from shapiq.vision import PatchStrategy, ViTClassificationArchitecture

vit_id = "google/vit-base-patch16-224"
vit_processor = AutoImageProcessor.from_pretrained(vit_id)
vit = AutoModelForImageClassification.from_pretrained(vit_id).eval()

architecture = ViTClassificationArchitecture(
    model=vit,
    vit_processor=vit_processor,
    player_strategy=PatchStrategy(grid_size=14, n_players=49),
)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
interaction_values = explainer.explain(budget=256)
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title("PatchStrategy (49 players)")
plt.show()
