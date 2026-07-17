"""
Architectures: Pixel and Token Masking
======================================

Every :class:`~shapiq.vision.ImageExplainer` is backed by an *architecture*
that bundles three things: the model, a player strategy (how the image is
split into regions), and a masking strategy (how removed regions are
replaced). You choose the architecture explicitly and pass it as the ``model``
argument of the explainer. Two exist:

- :class:`~shapiq.vision.ClassificationArchitecture` operates in **pixel
  space**: masking edits the image itself. It accepts any classification
  model -- CNNs, torchvision ViTs, or Hugging Face models (pass the
  ``processor``).
- :class:`~shapiq.vision.ViTClassificationArchitecture` operates in **token
  space**: masking removes patch tokens before the forward pass. It requires
  a Hugging Face ViT that honors ``bool_masked_pos``.

Both take optional ``player_strategy`` and ``masking_strategy`` arguments;
this example constructs each with and without overrides.
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
from transformers import AutoImageProcessor, AutoModelForImageClassification

hf_logging.set_verbosity_error()  # hide hub warnings (e.g. unauthenticated requests)
transformers.logging.set_verbosity_error()  # hide download noise and load reports
transformers.utils.logging.disable_progress_bar()

resize_and_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
pil_image = Image.open(Path("imagenet_sample.png")).convert("RGB")
image = np.array(resize_and_crop(pil_image))

# %%
# ClassificationArchitecture: Pixel Masking
# -----------------------------------------
# Pass just a model and it defaults to
# :class:`~shapiq.vision.SuperpixelStrategy` players and
# :class:`~shapiq.vision.MeanColorMasking`. Override either to change the
# unit of explanation or the removal rule -- here a 4x4 grid with zero
# masking. As in the quickstart, the ImageNet normalization lives inside the
# model because the pixel path feeds images scaled to ``[0, 1]``.

from shapiq.vision import ClassificationArchitecture, ImageExplainer, ZeroMasking
from shapiq.vision.players import GridStrategy

resnet = torch.nn.Sequential(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
)
resnet = resnet.eval()

architecture = ClassificationArchitecture(
    model=resnet,
    player_strategy=GridStrategy(grid_shape=4),
    masking_strategy=ZeroMasking(),
)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
interaction_values = explainer.explain(budget=256)
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title("ClassificationArchitecture: explicit players and masking")
plt.show()

# %%
# ViTClassificationArchitecture: Token Masking
# --------------------------------------------
# A Hugging Face ViT is explained in token space. The processor is required
# (it produces the ``pixel_values`` the model expects); players and masking
# default to :class:`~shapiq.vision.PatchStrategy` (sized to the token grid)
# and :class:`~shapiq.vision.MaskTokenStrategy`.

from shapiq.vision import ViTClassificationArchitecture

vit_id = "google/vit-base-patch16-224"
vit_processor = AutoImageProcessor.from_pretrained(vit_id)
vit = AutoModelForImageClassification.from_pretrained(vit_id).eval()

architecture = ViTClassificationArchitecture(model=vit, vit_processor=vit_processor)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
interaction_values = explainer.explain(budget=16)  # 4 default players -> exact
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title("ViTClassificationArchitecture: token masking")
plt.show()

# %%
# The constructors validate their configuration: pairing a pixel-space player
# or masking strategy with ``ViTClassificationArchitecture`` (or a token-space
# one with ``ClassificationArchitecture``) raises a ``TypeError`` immediately,
# rather than producing meaningless attributions.
#
# Forcing Pixel Masking for a Transformer
# ---------------------------------------
# A ViT does not have to be explained in token space. Constructing
# :class:`~shapiq.vision.ClassificationArchitecture` with the model's
# ``processor`` masks the same ViT in pixel space -- useful when every model
# in a comparison should play the same game, or for a model whose head
# ignores ``bool_masked_pos``.

architecture = ClassificationArchitecture(model=vit, processor=vit_processor)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
interaction_values = explainer.explain(budget=128)
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title("ViT on the pixel path (superpixel players)")
plt.show()
