"""
Masking Strategies
==================

``shapiq.vision`` selects the masking strategy with the ``masking_strategy``
argument of the architecture.

Pixel-space strategies replace removed pixels and are used with
:class:`~shapiq.vision.ClassificationArchitecture`:

- :class:`~shapiq.vision.MeanColorMasking` (default) -- fills removed
  regions with the image's per-channel mean color.
- :class:`~shapiq.vision.ZeroMasking` -- fills removed regions with a
  constant, ``ZeroMasking(value=0.0)`` by default.
- :class:`~shapiq.vision.BlurMasking` -- a Gaussian-blurred copy of the image.
- :class:`~shapiq.vision.DatasetMeanMasking` -- a fixed dataset-wide mean color.
- :class:`~shapiq.vision.MarginalSampling` -- pixels drawn from a bank of
  reference images (the marginal removal of :footcite:t:`Covert.2021b`).
- :class:`~shapiq.vision.InpaintingMasking` -- a user-supplied inpainter (the
  conditional removal of :footcite:t:`Covert.2021b`).

This example demonstrates the first two; the rest share the same interface.

Token-space strategies remove patch tokens before the forward pass and are
used with :class:`~shapiq.vision.ViTClassificationArchitecture`:

- :class:`~shapiq.vision.MaskTokenStrategy` (default) -- replaces removed
  tokens with an all-zero mask-token embedding, creating that embedding if
  the checkpoint ships none.
- :class:`~shapiq.vision.BoolMaskedPosStrategy` -- passes the token mask
  directly to the model, relying on the checkpoint's own trained mask token
  (available in masked-image-modeling models such as
  ``ViTForMaskedImageModeling``).

Mixing domains -- a pixel strategy on ``ViTClassificationArchitecture`` or a token
strategy on ``ClassificationArchitecture`` -- raises a ``TypeError`` at construction.
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
# Pixel-Space: What Each Strategy Does to the Image
# -------------------------------------------------
# Masking strategies can be applied standalone. Here is the same coalition
# -- only the four center patches of a 4x4 grid survive -- under both pixel
# strategies:

from shapiq.vision import MeanColorMasking, ZeroMasking
from shapiq.vision.players import GridStrategy

strategy = GridStrategy(grid_shape=4)
player_masks = torch.from_numpy(strategy.get_masks(image))
image_tensor = torch.from_numpy(image / 255.0).float().permute(2, 0, 1)

coalition = torch.zeros(1, 16, dtype=torch.bool)
coalition[0, [5, 6, 9, 10]] = True  # keep only the four center patches

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, masking in zip(axes, [MeanColorMasking(), ZeroMasking()], strict=True):
    masked = masking.apply(image_tensor, player_masks, coalition)[0]
    ax.imshow(masked.permute(1, 2, 0).numpy())
    ax.set_title(type(masking).__name__)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# Selecting a Pixel Strategy
# --------------------------
# Pass the strategy to :class:`~shapiq.vision.ClassificationArchitecture`. Different
# replacement rules define different games, so the attributions differ
# :footcite:t:`Covert.2021b`; everything else below is held fixed.

from shapiq.vision import ClassificationArchitecture, ImageExplainer

for masking in [MeanColorMasking(), ZeroMasking()]:
    architecture = ClassificationArchitecture(
        model=model,
        masking_strategy=masking,
        player_strategy=GridStrategy(grid_shape=4),
    )
    explainer = ImageExplainer(model=architecture, data=image, random_state=42)
    interaction_values = explainer.explain(budget=256)
    fig, ax = interaction_values.plot_image_attributions(
        image, explainer.imputer.player_masks, show=False
    )
    ax.set_title(type(masking).__name__)
    plt.show()

# %%
# Selecting a Token Strategy
# --------------------------
# Token strategies go to :class:`~shapiq.vision.ViTClassificationArchitecture` the
# same way. :class:`~shapiq.vision.MaskTokenStrategy` takes the model as its
# argument (it manages the model's mask-token embedding):

from transformers import AutoImageProcessor, AutoModelForImageClassification

from shapiq.vision import MaskTokenStrategy, ViTClassificationArchitecture

vit_id = "google/vit-base-patch16-224"
vit_processor = AutoImageProcessor.from_pretrained(vit_id)
vit = AutoModelForImageClassification.from_pretrained(vit_id).eval()

architecture = ViTClassificationArchitecture(
    model=vit,
    vit_processor=vit_processor,
    masking_strategy=MaskTokenStrategy(vit),
)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
interaction_values = explainer.explain(budget=16)  # 4 default players -> exact
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title("MaskTokenStrategy")
plt.show()

# %%
# Use :class:`~shapiq.vision.BoolMaskedPosStrategy` instead when the
# checkpoint carries its own trained mask token (masked-image-modeling
# models); it forwards the token mask without touching the embedding:
# ``ViTClassificationArchitecture(model=model, vit_processor=processor,
# masking_strategy=BoolMaskedPosStrategy())``.
#
# Practical guidance for the pixel strategies: mean color keeps masked
# inputs closer to natural image statistics, while black patches are further
# out of distribution and a CNN can react to their hard edges as if they
# were content. Whatever you pick, keep it fixed when comparing
# explanations, because switching the removal rule switches the game.

# %%
# References
# ----------
# .. footbibliography::
