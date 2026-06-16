"""Segmenter base — abstract contract and configuration for spatial division strategies.

Defines:
    - Per-strategy parameter dataclasses: PatchParams, SlicParams,
      CustomSegmenterParams
    - SegmenterConfig — caller-provided + factory-populated metadata
    - Segmenter(ABC) — abstract base for all segmenters
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

    from shapiq.imputer.vision.base import PhysicalMask, SpatialLayout

# ═══════════════════════════════════════════════════════════════════════
# Per-strategy parameter dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PatchParams:
    """Rigid-grid patch segmenter parameters.  No configurable knobs."""


@dataclass
class SlicParams:
    """SLIC superpixel segmentation parameters.

    Attributes:
        n_segments: Target superpixel count (default 49, ~ 7x7 grid).
        compactness: SLIC compactness factor (higher = more regular shapes).
        sigma: Pre-segmentation Gaussian blur sigma.
    """

    n_segments: int = 49
    compactness: float = 10.0
    sigma: float = 0.0


@dataclass
class CustomSegmenterParams:
    """User-provided binary mask segmenter parameters.

    Each mask defines one player. Masks are supplied directly to
    ``CustomSegmenter`` at construction time, not via this dataclass.
    """


# ═══════════════════════════════════════════════════════════════════════
# Segmenter Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SegmenterConfig:
    """Complete configuration for a Segmenter.

    Caller-provided: strategy + per-strategy params (patch / slic ).
    Factory-populated: model metadata (image_size, patch_size, model_type, ...).
    Default strategy is ``"patch"``.
    """

    strategy: str = "patch"
    patch: PatchParams = field(default_factory=PatchParams)
    slic: SlicParams = field(default_factory=SlicParams)
    custom_segmenter: CustomSegmenterParams = field(default_factory=CustomSegmenterParams)

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
    def active_params(self) -> object:
        """Return the active configuration dataclass based on strategy name."""
        return getattr(self, self.strategy, None)


# ═══════════════════════════════════════════════════════════════════════
# Abstract Segmenter
# ═══════════════════════════════════════════════════════════════════════


class Segmenter(ABC):
    """Abstract base class for spatial division strategies.

    A Segmenter defines which pixels/tokens belong to which player.
    It produces a SpatialLayout and converts coalition arrays into
    PhysicalMask tensors.
    """

    def __init__(self, config: SegmenterConfig) -> None:
        """Initialize the segmenter with its configuration."""
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
        coalitions_image: np.ndarray | None = None,
        coalitions_text: np.ndarray | None = None,
        device: torch.device | None = None,
    ) -> PhysicalMask:
        """Translate boolean coalition arrays into concrete physical masks.

        Args:
            coalitions_image: (N_img, n_players_image) bool.
            coalitions_text: (N_txt, n_players_text) bool.
            device: Target device for the generated masks.

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
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Convert token-level coalitions to attention masks.

        Builds an attention mask matching the model's expected total length
        based on model_type (CLIP: bos/eos padding, SigLIP: trailing ones).

        Args:
            coalitions: (N, n_players_text) bool.
            device: Target device for the generated mask.

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
        msg = f"Unsupported model_type: {cfg.model_type}"
        raise ValueError(msg)
