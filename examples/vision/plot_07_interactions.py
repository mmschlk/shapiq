"""
Computing Shapley Interactions for Images
=========================================

Shapley values score each region alone; Shapley *interactions* additionally
score pairs (and higher orders) of regions. Request them by passing the
k-Shapley Interaction Index (``index="k-SII"``) and ``max_order=2`` to
:class:`~shapiq.vision.ImageExplainer`. This example computes exact pairwise
interactions on a 3x3 grid and shows both visualizations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# %%
# Image and model as in the quickstart -- the ImageNet normalization lives
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
# Compute Exact Interactions
# --------------------------
# The players come from a fixed 3x3
# :class:`~shapiq.vision.players.GridStrategy` (see
# :ref:`sphx_glr_auto_examples_vision_plot_04_player_strategies.py` for the player
# strategies). A 3x3 grid keeps the game small: 9 players means
# :math:`2^9 = 512` coalitions, so a budget of 512 evaluates every one of
# them and the values are exact.

from shapiq.vision import ClassificationArchitecture, ImageExplainer
from shapiq.vision.players import GridStrategy

architecture = ClassificationArchitecture(model=model, player_strategy=GridStrategy(grid_shape=3))
explainer = ImageExplainer(
    model=architecture, data=image, index="k-SII", max_order=2, random_state=42
)
interaction_values = explainer.explain(budget=512)
print(interaction_values)

# %%
# First-Order Values: Heatmap
# ---------------------------

interaction_values.plot_image_attributions(image, explainer.imputer.player_masks)

# %%
# Second-Order Values: Network Plot
# ---------------------------------
# The network plot shows both orders at once: nodes are the grid cells
# (numbered row-major from the top-left), edges are the pairwise
# interactions.

cell_names = [f"Cell {i}" for i in range(explainer.imputer.n_features)]
interaction_values.plot_network(feature_names=cell_names)

# %%
# Strongest Pairs
# ---------------

order2 = interaction_values.get_n_order_values(2)
n_players = explainer.imputer.n_features
pairs = sorted(
    (((i, j), order2[i, j]) for i in range(n_players) for j in range(i + 1, n_players)),
    key=lambda pair: -abs(pair[1]),
)
for (i, j), value in pairs[:5]:
    print(f"cells {i} & {j}: {value:+.4f}")

# %%
# A positive pair adds more together than their individual values suggest --
# synergy, typically two parts of the same object that only work as evidence
# in combination. A negative pair is redundancy: either region alone carries
# the information, so their joint credit is discounted.
