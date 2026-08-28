"""PatchSegmenter — Rigid-grid segmenter aligned with Vision Transformer embeddings.

Each patch is a single player. Default baseline for VLMs (CLIP, SigLIP)
since their vision encoders natively operate on patches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from shapiq.imputer.vision.base import PhysicalMask, SpatialLayout
from shapiq.imputer.vision.segmenters.base import Segmenter, SegmenterConfig

from . import register_segmenter

if TYPE_CHECKING:
    import numpy as np


@register_segmenter("patch")
class PatchSegmenter(Segmenter):
    """Rigid-grid segmenter with one player per ViT patch.

    Args:
        config: SegmenterConfig with strategy ``"patch"``. Model metadata
            (image_size, patch_size, grid_size) must already be populated
            by the Factory.
    """

    def __init__(self, config: SegmenterConfig) -> None:
        """Initialize the patch segmenter."""
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.n_channels = config.n_channels
        self.n_players_text = config.n_players_text
        self.model_type = config.model_type
        self.text_total_length = config.text_total_length
        self.grid_size = config.grid_size
        self.n_players_image = config.n_players_image

        self._layout = SpatialLayout(
            n_players_image=self.n_players_image,
            n_players_text=self.n_players_text,
            image_size=self.image_size,
            patch_size=self.patch_size,
            grid_size=self.grid_size,
            n_channels=self.n_channels,
            model_type=self.model_type,
            text_total_length=self.text_total_length,
            is_stateful=False,
        )

    def get_layout(self) -> SpatialLayout:
        """Return the spatial layout for this segmenter."""
        return self._layout

    def generate_masks(
        self,
        coalitions_image: np.ndarray | None = None,
        coalitions_text: np.ndarray | None = None,
        device: torch.device | None = None,
    ) -> PhysicalMask:
        """Generate physical masks from coalition arrays."""
        mask = PhysicalMask()
        if coalitions_image is not None:
            mask.image_binary_mask = self._generate_image_mask(coalitions_image, device=device)
        if coalitions_text is not None:
            mask.text_attention_mask = self._build_text_attention_mask(
                coalitions_text, device=device
            )
        return mask

    def _generate_image_mask(
        self,
        coalitions: np.ndarray,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Convert patch-level coalition array to pixel-level binary mask.

        Returns:
            Tensor (N, C, H, W) with 1=keep, 0=occlude.
        """
        coalition_t = torch.as_tensor(coalitions, dtype=torch.bool, device=device)
        n_coalitions = coalition_t.shape[0]
        binary_masks = coalition_t.repeat_interleave(self.patch_size**2, dim=1).reshape(
            n_coalitions, self.grid_size, self.grid_size, self.patch_size, self.patch_size
        )
        binary_masks = binary_masks.permute(0, 1, 3, 2, 4).reshape(
            n_coalitions, self.image_size, self.image_size
        )
        binary_masks = binary_masks.unsqueeze(1).repeat(1, self.n_channels, 1, 1)
        return binary_masks.float()
