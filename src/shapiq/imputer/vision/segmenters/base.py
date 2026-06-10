"""
Segmenter base — abstract contract and configuration for spatial division strategies.

Defines:
    - Per-strategy parameter dataclasses: PatchParams, SlicParams,
      GradientGuidedParams, CustomSegmenterParams
    - SegmenterConfig — caller-provided + factory-populated metadata
    - Segmenter(ABC) — abstract base for all segmenters
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch

from ..base import PhysicalMask, SpatialLayout


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

    n_segments: int | None = None


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

    Caller-provided: strategy + per-strategy params (patch / slic / gradient_guided).
    Factory-populated: model metadata (image_size, patch_size, model_type, ...).
    Default strategy is ``"patch"``.
    """

    strategy: str = "patch"
    patch: PatchParams = field(default_factory=PatchParams)
    slic: SlicParams = field(default_factory=SlicParams)
    gradient_guided: GradientGuidedParams = field(default_factory=GradientGuidedParams)
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
    def active_params(self):
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
        coalitions_image: np.ndarray | None = None,
        coalitions_text: np.ndarray | None = None,
        device: torch.device | None = None,
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
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Convert token-level coalitions to an attention mask matching the
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
