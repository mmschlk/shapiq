"""
Explaining a Class of Your Choice
=================================

By default :class:`~shapiq.vision.ImageExplainer` explains the model's top
prediction. To explain any other class, pass its index as ``class_index`` --
everything else about the game stays identical. This example shows how to
find the index of the class you care about, and uses it to ask the model two
different questions about the same image.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# %%
# Image and model as in the quickstart -- the ImageNet normalization lives
# inside the model because ``shapiq`` feeds it images scaled to ``[0, 1]``
# (see :ref:`sphx_glr_auto_examples_vision_plot_01_quickstart_cnn.py`). The
# sample image contains a cat *and* a dog, but the model's top predictions
# are all cat classes:

resize_and_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
pil_image = Image.open(Path("imagenet_sample.png")).convert("RGB")
image = np.array(resize_and_crop(pil_image))

weights = models.ResNet18_Weights.IMAGENET1K_V1
categories = weights.meta["categories"]
model = torch.nn.Sequential(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    models.resnet18(weights=weights),
)
model = model.eval()

x = torch.from_numpy(image / 255.0).float().permute(2, 0, 1).unsqueeze(0)
with torch.no_grad():
    probs = model(x).softmax(-1)[0]
top = probs.topk(5)
for p, i in zip(top.values, top.indices, strict=True):
    print(f"{categories[int(i)]:24s} {p:.1%}")

# %%
# Finding the Index of Your Class
# -------------------------------
# ``class_index`` is the position of the class in the model's output vector.
# torchvision ships the label list with the weights, so the index of any
# label is one lookup away. (Hugging Face models carry the same mapping as
# ``model.config.label2id``.) The dog is nowhere near the top of the
# ranking, so it is a class we have to ask about explicitly:

class_dog = categories.index("collie")
print(f"'collie' is class {class_dog}, currently at {probs[class_dog]:.1%}")

# %%
# Explain the Class of Your Choice
# --------------------------------
# Pass the index as ``class_index``. The game -- players, masking, budget --
# is unchanged; only the score being explained is different.

from shapiq.vision import ClassificationArchitecture, ImageExplainer

architecture = ClassificationArchitecture(model=model)
explainer = ImageExplainer(model=architecture, data=image, class_index=class_dog, random_state=42)
interaction_values = explainer.explain(budget=256)
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title(f"Evidence for '{categories[class_dog]}'")
plt.show()

# %%
# Compare Against the Default
# ---------------------------
# Omitting ``class_index`` explains the top prediction, here ``tiger cat``.
# Because everything else is identical, the two heatmaps are directly
# comparable.

explainer = ImageExplainer(
    model=ClassificationArchitecture(model=model), data=image, random_state=42
)
interaction_values = explainer.explain(budget=256)
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title(f"Evidence for '{categories[int(probs.argmax())]}'")
plt.show()

# %%
# Regions that are red in one map and blue in the other are the
# discriminating evidence between the two classes; regions with the same
# color in both support (or hurt) them jointly.
