"""
Base module for the vision imputer sub-package — data transfer protocol.

Defines data types shared by all sub-modules:
    - SpatialLayout — describes how input is divided into players
    - PhysicalMask — concrete pixel/token-level masks
    - ProcessorOutput — standardised HuggingFace processor wrapper

Segmenter and Masker abstract contracts now live in:
    - shapiq.imputer.vision.segmenters.base
    - shapiq.imputer.vision.maskers.base
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ═══════════════════════════════════════════════════════════════════════
# Spatial Layout
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SpatialLayout:
    """Describes the spatial division of the input into players.

    Attributes:
        n_players_image: Number of image players (patches/superpixels).
        n_players_text: Number of text players (tokens after removing BOS/EOS).
        image_size: Height/width of the input image in pixels.
        patch_size: Edge length of a single patch.
        grid_size: Number of patches per side (image_size // patch_size).
        n_channels: Number of image channels (typically 3).
        model_type: 'clip', 'siglip', or 'siglip2'.
        text_total_length: Total token length expected by the model.
        is_stateful: Whether the layout can change across iterations.
    """

    n_players_image: int
    n_players_text: int
    image_size: int
    patch_size: int
    grid_size: int
    n_channels: int
    model_type: str
    text_total_length: int
    is_stateful: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Physical Mask
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PhysicalMask:
    """Concrete, pixel/token-level masks ready to be applied to model inputs.

    Attributes:
        image_binary_mask: Tensor (N_img, C, H, W) float/bool.
            1 = keep, 0 = occlude.
        text_attention_mask: Tensor (N_txt, L) int.
            1 = attend, 0 = ignore. Already padded for model_type.
    """

    image_binary_mask: torch.Tensor | None = None
    text_attention_mask: torch.Tensor | None = None

    @property
    def batch_size_img(self) -> int:
        return self.image_binary_mask.shape[0] if self.image_binary_mask is not None else 0

    @property
    def batch_size_txt(self) -> int:
        return self.text_attention_mask.shape[0] if self.text_attention_mask is not None else 0


# ═══════════════════════════════════════════════════════════════════════
# Processor Output
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ProcessorOutput:
    """Standardised wrapper around HuggingFace processor outputs.

    Attributes:
        pixel_values: Tensor (B, C, H, W).
        input_ids: Tensor (B, L).
        attention_mask: Tensor (B, L).
        model_type: str identifying the model family.
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    model_type: str

    @classmethod
    def from_hf_processor(cls, inputs: dict, model_type: str) -> ProcessorOutput:
        """Create from a HuggingFace processor output dict."""
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (inputs["input_ids"] != 1).long()
        return cls(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            model_type=model_type,
        )

    def to_dict(self) -> dict:
        return {
            "pixel_values": self.pixel_values,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

    @property
    def device(self):
        return self.pixel_values.device

    def to(self, device):
        self.pixel_values = self.pixel_values.to(device)
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self
