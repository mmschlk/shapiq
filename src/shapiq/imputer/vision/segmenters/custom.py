"""CustomSegmenter — User-provided binary masks as players.

Each user-supplied binary mask defines one player. Mask shape:
``(n_players, H, W)`` where ``H`` and ``W`` match the image's spatial
dimensions after preprocessing.

Useful for:
    - Semantic / instance segmentation masks
    - Region-of-interest analysis
    - Comparing learned segmenters against ground-truth regions
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..base import PhysicalMask, SpatialLayout
from . import register_segmenter
from .base import Segmenter, SegmenterConfig


@register_segmenter("custom_segmenter")
class CustomSegmenter(Segmenter):
    """Segmenter that uses user-provided binary masks directly as players.

    Args:
        config: ``SegmenterConfig`` with strategy ``"custom_segmenter"``.
        masks: Binary mask array of shape ``(n_players, H, W)``.
            Accepted types: ``np.ndarray``, ``torch.Tensor``, or any
            type that ``np.asarray`` can convert. Values are treated as
            bool (non-zero = player pixel).

    Raises:
        ValueError: If ``masks`` is not 3D, or if the mask count does
            not match the number of players implied by the config.
    """

    def __init__(self, config: SegmenterConfig, masks: Any = None) -> None:  # noqa: ANN401
        """Initialize the segmenter with user-provided masks."""
        super().__init__(config)
        if masks is None:
            msg = "CustomSegmenter requires ``masks`` of shape (N_players, H, W)."
            raise ValueError(msg)

        masks_arr = np.asarray(masks, dtype=bool)
        if masks_arr.ndim != 3:
            msg = (
                f"``masks`` must be a 3D array (N_players, H, W), "
                f"got shape {masks_arr.shape} and ndim={masks_arr.ndim}."
            )
            raise ValueError(msg)

        self.n_players_image = int(masks_arr.shape[0])
        self.image_size = int(masks_arr.shape[1])
        if masks_arr.shape[2] != self.image_size:
            msg = (
                f"``masks`` must have square spatial dimensions (H, W) with H == W, "
                f"got {masks_arr.shape[1]}x{masks_arr.shape[2]}."
            )
            raise ValueError(msg)

        self.n_channels = config.n_channels or 3
        self.n_players_text = config.n_players_text
        self.model_type = config.model_type or "clip"
        self.text_total_length = config.text_total_length or 0

        # Store mask tensors on CPU; move to target device on demand
        self._masks = torch.from_numpy(masks_arr).bool()  # (K, H, W) bool
        self._masks_by_device: dict = {torch.device("cpu"): self._masks}

        self._layout = SpatialLayout(
            n_players_image=self.n_players_image,
            n_players_text=self.n_players_text,
            image_size=self.image_size,
            patch_size=0,
            grid_size=0,
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
        """Convert player coalitions to pixel-level physical masks."""
        mask = PhysicalMask()
        if coalitions_image is not None:
            mask.image_binary_mask = self._scatter_image_mask(
                coalitions_image,
                device=device,
            )
        if coalitions_text is not None:
            mask.text_attention_mask = self._build_text_attention_mask(
                coalitions_text,
                device=device,
            )
        return mask

    def _scatter_image_mask(
        self,
        coalitions: np.ndarray,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Convert coalition array to pixel-level binary mask.

        For each coalition row, the pixel mask is the **union** of all
        selected players' masks: a pixel is 1 (keep) if *any* selected
        player covers it.

        Args:
            coalitions: (N, K) bool array.
            device: Target device for the output tensor.

        Returns:
            Float tensor (N, C, H, W) with 1 = keep, 0 = occlude.
        """
        coalition_t = torch.as_tensor(coalitions, dtype=torch.bool, device=device)
        masks = self._masks_for(coalition_t.device)  # (K, H, W) bool

        # Union over selected players: (N, K) @ (K, H*W) → (N, H*W) → (N, H, W)
        K, H, W = masks.shape
        pixel_masks = (coalition_t.float() @ masks.reshape(K, -1).float()).reshape(
            -1, H, W
        )  # (N, H, W) float
        pixel_masks = (pixel_masks > 0).float()  # binarise union

        return pixel_masks.unsqueeze(1).expand(-1, self.n_channels, -1, -1)

    def _masks_for(self, device: torch.device) -> torch.Tensor:
        """Return cached mask tensor on the requested device."""
        device = torch.device(device)
        cached = self._masks_by_device.get(device)
        if cached is None:
            cached = self._masks.to(device=device, non_blocking=True)
            self._masks_by_device[device] = cached
        return cached
