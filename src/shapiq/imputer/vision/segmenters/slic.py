"""SLICSegmenter — Perceptual superpixel segmenter (skimage SLIC).

Algorithm:
    1. __init__ (CPU planning, once): run skimage.segmentation.slic on the
       raw input image. Result is a (H, W) integer label map; each label
       defines one player.
    2. generate_masks (per batch): for each coalition row (N, K) bool,
       gather labels into a (N, H, W) pixel mask via fancy-indexing.

Use case: CNN-backbone CLIP variants (RN50/RN101/RN50x4). Rigid patch grids
would introduce out-of-distribution artefacts; SLIC follows perceptual
content boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from shapiq.imputer.vision.base import PhysicalMask, SpatialLayout
from shapiq.imputer.vision.segmenters.base import Segmenter, SegmenterConfig

from . import register_segmenter

if TYPE_CHECKING:
    import PIL.Image

try:
    from skimage.segmentation import slic as _skimage_slic
except ImportError:
    _skimage_slic = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore[assignment]


@register_segmenter("slic")
class SLICSegmenter(Segmenter):
    """Perceptual superpixel segmenter for VLMs (especially CNN backbones).

    Args:
        config: SegmenterConfig with strategy ``"slic"``.
        image_array: Raw input image (PIL.Image or np.ndarray). The Factory
            passes this automatically when strategy="slic".

    Strategy parameters (via config.params):
        n_segments (int): target superpixel count (default 49).
        compactness (float): SLIC compactness (default 10.0).
        sigma (float): pre-smoothing Gaussian sigma (default 0.0).
    """

    def __init__(
        self,
        config: SegmenterConfig,
        image_array: PIL.Image.Image | np.ndarray | None = None,
    ) -> None:
        """Initialize the SLIC segmenter."""
        super().__init__(config)
        if _skimage_slic is None:
            msg = "SLICSegmenter requires scikit-image."
            raise ImportError(msg)

        self.image_size = config.image_size
        self.n_channels = config.n_channels
        self.n_players_text = config.n_players_text
        self.model_type = config.model_type
        self.text_total_length = config.text_total_length

        if image_array is None:
            msg = "SLICSegmenter requires image_array."
            raise ValueError(msg)

        n_segments = int(config.params.n_segments)
        compactness = float(config.params.compactness)
        sigma = float(config.params.sigma)

        # CPU planning (once)
        image_rgb = self._coerce_rgb_uint8(image_array, self.image_size)
        raw_labels = _skimage_slic(
            image_rgb,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0,
            channel_axis=-1,
        )
        unique_ids, packed = np.unique(raw_labels, return_inverse=True)
        label_map = packed.reshape(self.image_size, self.image_size).astype(np.int64)
        self.n_players_image = int(unique_ids.size)
        self._label_map = torch.from_numpy(label_map)  # CPU (H, W) int64
        self._label_map_by_device = {torch.device("cpu"): self._label_map}

        self._layout = SpatialLayout(
            n_players_image=self.n_players_image,
            n_players_text=self.n_players_text,
            image_size=self.image_size,
            patch_size=0,  # N/A for SLIC
            grid_size=0,  # N/A for SLIC
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
            mask.image_binary_mask = self._scatter_image_mask(coalitions_image, device=device)
        if coalitions_text is not None:
            mask.text_attention_mask = self._build_text_attention_mask(
                coalitions_text, device=device
            )
        return mask

    def _scatter_image_mask(
        self,
        coalitions: np.ndarray,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Translate (N, K) bool coalitions → (N, C, H, W) float pixel masks."""
        coalition_t = torch.as_tensor(coalitions, dtype=torch.bool, device=device)
        label_map = self._label_map_for(coalition_t.device)
        pixel_masks = coalition_t[:, label_map]  # (N, H, W)
        return pixel_masks.unsqueeze(1).expand(-1, self.n_channels, -1, -1).float()

    def _label_map_for(self, device: torch.device) -> torch.Tensor:
        device = torch.device(device)
        cached = self._label_map_by_device.get(device)
        if cached is None:
            cached = self._label_map.to(device=device, non_blocking=True)
            self._label_map_by_device[device] = cached
        return cached

    @staticmethod
    def _coerce_rgb_uint8(image: PIL.Image.Image | np.ndarray, target_size: int) -> np.ndarray:
        """Normalise PIL.Image / ndarray inputs to (H, W, 3) uint8 at target_size."""
        if PILImage is not None and isinstance(image, PILImage.Image):
            return np.asarray(
                image.convert("RGB").resize((target_size, target_size)), dtype=np.uint8
            )

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr_max = float(arr.max()) if arr.size > 0 else 0.0
            arr = (arr * 255 if arr_max <= 1.0 else arr).clip(0, 255).astype(np.uint8)
        if arr.shape[:2] != (target_size, target_size):
            if PILImage is None:
                msg = "PIL is required to resize ndarray inputs for SLIC."
                raise ImportError(msg)
            arr = np.asarray(
                PILImage.fromarray(arr).resize((target_size, target_size)), dtype=np.uint8
            )
        return arr
