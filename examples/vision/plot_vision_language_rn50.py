"""
Vision-Language Explanation with CLIP RN50 + SLIC Superpixels
===============================================================

This example shows how to explain a **CNN-backbone CLIP model** (RN50) using
SLIC superpixels instead of rigid patches. Unlike ViT models, CNN-based CLIP
variants cannot use the patch segmenter — the grid does not align with the
model's internal representations.

**Pipeline:**

1. **SLIC Segmenter** divides the image into perceptually meaningful superpixels.
2. **CrossModalMeanMasker** applies zero-out occlusion to image + text players.
3. :class:`~shapiq.explainer.vision.VisionLanguageExplainer` wraps the pipeline
   and computes interaction values.

**Model:** OpenAI CLIP RN50 — 49 SLIC superpixels.

Requires the OpenAI ``clip`` package (``pip install git+https://github.com/openai/CLIP.git``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import clip
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from shapiq.explainer.vision import VisionLanguageExplainer
from shapiq.imputer.vision import (
    MaskerConfig,
    SegmenterConfig,
    SlicParams,
)

# ─── OpenAI CLIP adapters ─────────────────────────────────────────────


class OpenAIClipBatch(dict):
    """Small BatchEncoding-like dict with a .tokens() helper for plotting."""

    def tokens(self) -> list[str]:
        n_text_players = int(self["input_ids"].shape[1]) - 2
        return ["<s>"] + [f"tok_{i}" for i in range(n_text_players)] + ["</s>"]


class OpenAICLIPProcessorAdapter:
    """Processor adapter that returns pixel_values, input_ids, and attention_mask."""

    def __init__(self, preprocess: object, context_length: int = 77) -> None:
        self.preprocess = preprocess
        self.context_length = int(context_length)
        self.image_processor = SimpleNamespace(
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
        )
        # tokenizer stub for .tokenizer.convert_ids_to_tokens API
        self.tokenizer = SimpleNamespace(
            convert_ids_to_tokens=lambda ids: [f"tok_{i}" for i in range(len(ids))]
        )

    @staticmethod
    def _as_list(value: object) -> list[object]:
        if isinstance(value, list | tuple):
            return list(value)
        return [value]

    def __call__(
        self,
        images,
        text,
        return_tensors="pt",  # noqa: ARG002
        padding=True,  # noqa: ARG002, FBT002
        max_length=None,  # noqa: ARG002
    ) -> OpenAIClipBatch:
        images = self._as_list(images)
        texts = self._as_list(text)

        pixel_values = torch.stack(
            [
                self.preprocess(
                    img.convert("RGB")
                    if hasattr(img, "convert")
                    else Image.fromarray(np.asarray(img)).convert("RGB")
                )
                for img in images
            ]
        )

        full_tokens = clip.tokenize(texts, context_length=self.context_length, truncate=True)
        lengths = []
        for row in full_tokens:
            nonzero = torch.nonzero(row, as_tuple=False).flatten()
            lengths.append(int(nonzero[-1].item()) + 1 if len(nonzero) else 2)
        short_len = max(lengths)
        input_ids = torch.zeros((len(texts), short_len), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), short_len), dtype=torch.long)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = full_tokens[i, :length]
            attention_mask[i, :length] = 1

        return OpenAIClipBatch(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class OpenAICLIPModelAdapter(torch.nn.Module):
    """HF-like wrapper around an official OpenAI CLIP model."""

    def __init__(self, clip_model: object, name_or_path: str) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.name_or_path = name_or_path
        self.context_length = int(getattr(clip_model, "context_length", 77))
        image_size = int(getattr(clip_model.visual, "input_resolution", 224))

        self.config = SimpleNamespace(
            model_type="clip",
            vision_config=SimpleNamespace(image_size=image_size, num_channels=3),
        )
        # CNN-style signal for VisionImputerFactory: no embeddings.patch_size.
        self.vision_model = SimpleNamespace()

    def forward(self, pixel_values, input_ids, attention_mask) -> SimpleNamespace:
        token_ids = input_ids.clone()
        token_ids = token_ids.masked_fill(attention_mask.to(token_ids.device) == 0, 0)

        if token_ids.shape[1] < self.context_length:
            pad = torch.zeros(
                (token_ids.shape[0], self.context_length - token_ids.shape[1]),
                dtype=token_ids.dtype,
                device=token_ids.device,
            )
            token_ids = torch.cat([token_ids, pad], dim=1)
        elif token_ids.shape[1] > self.context_length:
            token_ids = token_ids[:, : self.context_length]

        logits_per_image, logits_per_text = self.clip_model(pixel_values, token_ids)
        return SimpleNamespace(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
        )


# %%
# 1. Load Model & Processor
# --------------------------
# Uses the official OpenAI ``clip`` package with adapter wrappers.

rn50_model, preprocess = clip.load("RN50", jit=False)
model = OpenAICLIPModelAdapter(rn50_model, name_or_path="openai/clip-rn50")
processor = OpenAICLIPProcessorAdapter(preprocess, context_length=model.context_length)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# %%
# 2. Load Input Image & Text
# ----------------------------
# Uses the bundled sample image ``tests/shapiq/data/dog_and_hydrant.png``.
# Replace with your own image path for other experiments.

INPUT_TEXT = "black dog next to a yellow hydrant"

image_path = Path("tests") / "shapiq" / "data" / "dog_and_hydrant.png"
if not image_path.exists():
    image_path = Path("../../tests/shapiq/data/dog_and_hydrant.png")

image = Image.open(image_path).convert("RGB")
print(f"Image size: {image.size}")

# %%
# 3. Build the Explainer with SLIC Segmenter
# --------------------------------------------
# CNN-backbone models use SLIC superpixels instead of rigid patch grids.
# :class:`~shapiq.explainer.vision.VisionLanguageExplainer` wraps the
# full pipeline — image and text are passed at **explain time**.

seg_cfg = SegmenterConfig(
    strategy="slic",
    params=SlicParams(n_segments=49, compactness=10.0, sigma=1.0),
)

msk_cfg = MaskerConfig(strategy="crossmodal_mean")

explainer = VisionLanguageExplainer(
    model=model,
    processor=processor,
    segmenter_config=seg_cfg,
    masker_config=msk_cfg,
    batch_size=64,
    index="k-SII",
    max_order=2,
    random_state=42,
)

iv = explainer.explain(
    x={"image": image, "text": INPUT_TEXT},
    budget=2**12,
)

game = explainer.game
imputer = explainer.game.imputer

print(f"Model type:           {imputer.model_type}")
print(f"Image players:        {game.n_players_image}")
print(f"Text players:         {game.n_players_text}")
print(f"Total players:        {game.n_players}")
print(f"Empty coalition:      {game.empty_value:.4f}")
print(f"Full coalition:       {game.full_value:.4f}")
print(iv)

# %%
# 4a. SLIC Superpixel Visualisation
# -----------------------------------
# Show the superpixel boundaries with segment labels and the tokenized text.

from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

# Decode token IDs to readable text tokens
text_tokens_raw = imputer.inputs_raw["input_ids"]
text_tokens_decoded = processor.tokenizer.convert_ids_to_tokens(text_tokens_raw[0].tolist())
text_tokens = text_tokens_decoded[1:-1]  # strip BOS/EOS placeholder tokens

# Read the SLIC label map from the segmenter
slic_seg = imputer.segmenter
label_map = slic_seg._label_map.cpu().numpy()  # (H, W) int64
n_segments = game.n_players_image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# -- Left panel: superpixel boundaries with segment labels --
image_np = np.array(image.resize((224, 224)))
ax1.imshow(image_np)

# Draw thin boundaries in white
boundaries = find_boundaries(label_map, mode="outer")
ax1.imshow(boundaries, cmap="gray", alpha=0.4)

# Place each segment label at the interior point furthest from the boundary
for seg_id in range(n_segments):
    mask = label_map == seg_id
    if mask.sum() == 0:
        continue
    dist = distance_transform_edt(mask)
    interior_pt = np.unravel_index(dist.argmax(), dist.shape)
    cy, cx = interior_pt
    ax1.text(
        cx,
        cy,
        str(seg_id),
        ha="center",
        va="center",
        fontsize=7,
        color="white",
        fontweight="bold",
        bbox={"boxstyle": "circle,pad=0.15", "facecolor": "black", "alpha": 0.6},
    )

ax1.set_title(f"SLIC Superpixels ({n_segments} segments)")
ax1.axis("off")

# -- Right panel: tokenized text display --
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
plt.savefig("vision_rn50_slic_segments.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_rn50_slic_segments.png")

# %%
# 4b. Force Plot (First-Order)
# ------------------------------

feature_names = [f"S{i}" for i in range(game.n_players_image)] + text_tokens

iv_first_order = iv.get_n_order(1)
iv_first_order.plot_force(feature_names=feature_names)
plt.savefig("vision_rn50_slic_force.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_rn50_slic_force.png")

# %%
# 4c. Interaction Network (Second-Order)
# ----------------------------------------
# Blue edges = positive synergy, red edges = negative (diminishing).

iv.plot_network(
    feature_names=feature_names,
    draw_threshold=0.0,
)
plt.savefig("vision_rn50_slic_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vision_rn50_slic_network.png")

# %%
# Summary
# --------
# This example demonstrated the :class:`~shapiq.explainer.vision.VisionLanguageExplainer`
# with a CNN-backbone CLIP model and SLIC superpixels:
#
# | Step | Component | What it does |
# |---|---|---|
# | Load model | OpenAI ``clip`` | RN50 (ResNet-50) |
# | Explain | ``VisionLanguageExplainer`` | Wraps imputer + approximator |
# | Segment | ``SLICSegmenter`` | 49 superpixel players |
# | Mask | ``CrossModalMeanMasker`` | Zero-out pixels + text attention |
# | Approximate | ``KernelSHAPIQ`` (k-SII) | First- and second-order attributions |
#
# CNN-based CLIP models (RN50, RN101, RN50x4) benefit from SLIC superpixels
# because the rigid patch grid does not align with their convolutional feature maps.
