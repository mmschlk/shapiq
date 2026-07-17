"""
Quickstart: Explaining a CNN Prediction
=======================================

``shapiq`` explains an image classifier by playing a cooperative game: the
image is split into regions (the *players*), the model is evaluated on many
versions of the image with some regions removed (*coalitions* of players),
and each region receives a Shapley value -- its average contribution to the
class score.

This example walks the shortest path: one torchvision ResNet-18, one image,
one attribution heatmap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# %%
# Load the Image
# --------------
# We center-crop the bundled ImageNet sample to 224x224 and keep it as a
# plain uint8 array -- ``shapiq`` handles the conversion to a tensor.

resize_and_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
pil_image = Image.open(Path("imagenet_sample.png")).convert("RGB")
image = np.array(resize_and_crop(pil_image))

fig, ax = plt.subplots()
ax.imshow(image)
ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# Load the Model
# --------------
# One thing to know about the pixel-masking path: ``shapiq`` edits the image
# itself and feeds the model float tensors scaled to ``[0, 1]``. torchvision
# models expect ImageNet-normalized input, so the normalization has to live
# *inside* the model -- prepend it with :class:`torch.nn.Sequential`.

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = torch.nn.Sequential(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    models.resnet18(weights=weights),
)
model = model.eval()

x = torch.from_numpy(image / 255.0).float().permute(2, 0, 1).unsqueeze(0)
with torch.no_grad():
    probs = model(x).softmax(-1)[0]
top_class = int(probs.argmax())
print(f"top prediction: {weights.meta['categories'][top_class]} ({probs[top_class]:.1%})")

# %%
# Explain
# -------
# ``shapiq`` bundles the model with a *player strategy* and a *masking
# strategy* in an architecture. :class:`~shapiq.vision.ClassificationArchitecture`
# is the pixel-space one; pass just the model and it defaults to roughly 10
# SLIC superpixels and mean-color masking.
# :class:`~shapiq.vision.ImageExplainer` then estimates Shapley values from
# ``budget`` model evaluations, explaining the top predicted class by default.
# With 10 players there are :math:`2^{10}` possible coalitions; a budget at or
# above that evaluates all of them and makes the values exact.
#
# See :ref:`sphx_glr_auto_examples_vision_plot_03_architectures.py` for the
# architecture hierarchy and how to customise it.

from shapiq.vision import ClassificationArchitecture, ImageExplainer

architecture = ClassificationArchitecture(model=model)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
print(
    f"{type(explainer.imputer.architecture).__name__} with {explainer.imputer.n_features} players"
)

interaction_values = explainer.explain(budget=256)

# %%
# Visualize
# ---------
# Red regions push the score toward the predicted class, blue regions push
# away from it, and the intensity is the size of the contribution.

interaction_values.plot_image_attributions(image, explainer.imputer.player_masks)

# %%
# That is the whole loop: model and image in, attribution heatmap out.
