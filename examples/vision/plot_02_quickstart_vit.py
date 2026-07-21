"""
Quickstart: Explaining a Vision Transformer
===========================================

For a Vision Transformer (ViT), ``shapiq`` removes a region by dropping its
patch tokens: the embeddings of removed patches are replaced with a neutral
mask token (through the model's ``bool_masked_pos`` mechanism), so no pixel
information from that region enters the transformer. This token-space removal
is provided by :class:`~shapiq.vision.ViTClassificationArchitecture` (see
:ref:`sphx_glr_auto_examples_vision_plot_03_architectures.py` for the two
architectures and when to use each).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from huggingface_hub import logging as hf_logging
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

hf_logging.set_verbosity_error()  # hide hub warnings (e.g. unauthenticated requests)
transformers.logging.set_verbosity_error()  # hide download noise and load reports
transformers.utils.logging.disable_progress_bar()

# %%
# Load the Image
# --------------
# The Hugging Face processor handles resizing and normalization internally,
# so the image can stay a plain uint8 array.

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

model_id = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id).eval()

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print("top prediction:", model.config.id2label[int(logits.argmax())])

# %%
# Explain
# -------
# Wrap the model and its processor in a
# :class:`~shapiq.vision.ViTClassificationArchitecture`, which masks in token
# space. The default splits ViT-base's 14x14 token grid into 4 quadrant
# players, so :math:`2^4 = 16` coalitions cover the whole game and
# ``budget=16`` is exact. For finer-grained players see
# :ref:`sphx_glr_auto_examples_vision_plot_04_player_strategies.py`.

from shapiq.vision import ImageExplainer, ViTClassificationArchitecture

architecture = ViTClassificationArchitecture(model=model, vit_processor=processor)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
print(
    f"{type(explainer.imputer.architecture).__name__} with {explainer.imputer.n_features} players"
)

interaction_values = explainer.explain(budget=16)

# %%
# Visualize
# ---------
# Red regions push the score toward the predicted class, blue regions push
# away from it.

interaction_values.plot_image_attributions(image, explainer.imputer.player_masks)
