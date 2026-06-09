"""
Base module for the vision imputer sub-package.

Defines:
    - Data transfer types: SegmenterConfig, MaskerConfig, SpatialLayout,
      PhysicalMask, ProcessorOutput, and per-strategy params.
    - Abstract contracts: Segmenter(ABC), Masker(ABC).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════
# Segmenter Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PatchParams:
    """Rigid-grid patch segmenter parameters.  No configurable knobs."""
    pass


@dataclass
class SlicParams:
    """SLIC superpixel segmentation parameters.

    Attributes:
        n_segments: Target superpixel count (default 49, ≈ 7×7 grid).
        compactness: SLIC compactness factor (higher = more regular shapes).
        sigma: Pre-segmentation Gaussian blur sigma.
    """
    n_segments: int = 49
    compactness: float = 10.0
    sigma: float = 0.0


@dataclass
class GradientGuidedParams:
    """Gradient-guided saliency segmentation parameters.

    Attributes:
        n_segments: Target superpixel count. None means derive from
            grid_size (ViT) or fall back to 49.
    """
    n_segments: Optional[int] = None


@dataclass
class SegmenterConfig:
    """
    Complete configuration for a Segmenter.

    Caller-provided: strategy + per-strategy params (patch / slic / gradient_guided).
    Factory-populated: model metadata (image_size, patch_size, model_type, ...).
    Default strategy is ``"patch"``.
    """
    strategy: str = "patch"
    patch: PatchParams = field(default_factory=PatchParams)
    slic: SlicParams = field(default_factory=SlicParams)
    gradient_guided: GradientGuidedParams = field(default_factory=GradientGuidedParams)

    # Factory-populated (model metadata)
    model_type: str = ""
    image_size: int = 0
    patch_size: int = 0
    n_channels: int = 3
    grid_size: int = 0
    n_players_image: int = 0
    n_players_text: int = 0
    text_total_length: int = 0

    @property
    def active_params(self):
        return getattr(self, self.strategy, None)


# ═══════════════════════════════════════════════════════════════════════
# Masker Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CrossModalMeanParams:
    """Cross-modal occlusion (vision-mean + text-attention)."""
    pass


@dataclass
class CrossModalBlurParams:
    """Cross-modal occlusion (vision-blur + text-attention)."""
    pass


@dataclass
class VisionMeanParams:
    """Pure image occlusion via multiplicative binary mask."""
    pass


@dataclass
class VisionBlurParams:
    """Pure image occlusion via Gaussian blur."""
    sigma: float = 3.0


@dataclass
class TextAttentionParams:
    """Pure text occlusion via attention_mask replacement."""
    pass


@dataclass
class MaskerConfig:
    """
    Complete configuration for a Masker.

    Caller-provided: strategy + per-strategy params.
    Default strategy is ``"crossmodal_mean"``.
    """
    strategy: str = "crossmodal_mean"
    crossmodal_mean: CrossModalMeanParams = field(default_factory=CrossModalMeanParams)
    crossmodal_blur: CrossModalBlurParams = field(default_factory=CrossModalBlurParams)
    vision_mean: VisionMeanParams = field(default_factory=VisionMeanParams)
    vision_blur: VisionBlurParams = field(default_factory=VisionBlurParams)
    text_attn: TextAttentionParams = field(default_factory=TextAttentionParams)

    @property
    def active_params(self):
        return getattr(self, self.strategy, None)


# ═══════════════════════════════════════════════════════════════════════
# Spatial Layout
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SpatialLayout:
    """
    Describes the spatial division of the input into players.

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
    """
    Concrete, pixel/token-level masks ready to be applied to model inputs.

    Attributes:
        image_binary_mask: Tensor (N_img, C, H, W) float/bool.
            1 = keep, 0 = occlude.
        text_attention_mask: Tensor (N_txt, L) int.
            1 = attend, 0 = ignore. Already padded for model_type.
    """
    image_binary_mask: Optional[torch.Tensor] = None
    text_attention_mask: Optional[torch.Tensor] = None

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
    """
    Standardised wrapper around HuggingFace processor outputs.

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


# ═══════════════════════════════════════════════════════════════════════
# Abstract Contracts
# ═══════════════════════════════════════════════════════════════════════


class Segmenter(ABC):
    """
    Abstract base class for spatial division strategies.

    A Segmenter defines which pixels/tokens belong to which player.
    It produces a SpatialLayout and converts coalition arrays into
    PhysicalMask tensors.
    """

    def __init__(self, config: SegmenterConfig):
        self.config = config

    @abstractmethod
    def get_layout(self) -> SpatialLayout:
        """Produce spatial layout describing player↔pixel/token mapping.

        Called once per image. Must NOT access the GPU.
        """
        ...

    @abstractmethod
    def generate_masks(
        self,
        coalitions_image: Optional[np.ndarray] = None,
        coalitions_text: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> PhysicalMask:
        """Translate boolean coalition arrays into concrete physical masks.

        Args:
            coalitions_image: (N_img, n_players_image) bool.
            coalitions_text: (N_txt, n_players_text) bool.

        Returns:
            PhysicalMask with image_binary_mask and/or text_attention_mask.

        Note:
            MUST NOT access the GPU inside this method
            ("CPU Planning, GPU Execution").
        """
        ...

    # ── Shared text-mask helper for VLM segmenters ──────────────────────

    def _build_text_attention_mask(
        self,
        coalitions: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert token-level coalitions to an attention mask matching the
        model's expected total length.

        Args:
            coalitions: (N, n_players_text) bool.

        Returns:
            (N, text_total_length) int — 1=attend, 0=ignore.
        """
        cfg = self.config
        coalition_t = torch.as_tensor(coalitions, dtype=torch.bool, device=device)
        n_coalitions = coalition_t.shape[0]

        if cfg.model_type in ("siglip", "siglip2"):
            pad_len = cfg.text_total_length - cfg.n_players_text
            return torch.cat(
                (
                    coalition_t,
                    torch.ones(n_coalitions, pad_len, device=coalition_t.device),
                ),
                dim=1,
            ).int()
        if cfg.model_type == "clip":
            return torch.cat(
                (
                    torch.ones(n_coalitions, 1, device=coalition_t.device),
                    coalition_t,
                    torch.ones(n_coalitions, 1, device=coalition_t.device),
                ),
                dim=1,
            ).int()
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")


class Masker(ABC):
    """
    Abstract base class for feature occlusion strategies.

    A Masker applies a PhysicalMask to model inputs (ProcessorOutput)
    and returns modified inputs ready for model.forward().
    """

    def __init__(self, config: Optional[MaskerConfig] = None):
        self.config = config or MaskerConfig()

    @abstractmethod
    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """Apply the physical mask to processor outputs.

        Args:
            processor_output: Preprocessed model inputs.
            physical_mask: Concrete pixel/token masks to apply.

        Returns:
            Modified ProcessorOutput with occlusion applied.

        Note:
            Implementations MUST clone inputs before mutation.
        """
        ...
