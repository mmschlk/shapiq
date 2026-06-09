"""
VisionBlurMasker — Image occlusion via Gaussian blur in masked regions.

Operates exclusively on pixel_values. Must never touch input_ids or
attention_mask.

.. note::
    Skeleton only. Implementation tracked as Team B task B3.2.
    Currently passes through unchanged (no-op).

Design (CPU only, Phase 1):
    Convert pixel_values to CPU numpy, apply skimage.filters.gaussian
    per-channel, blend with mask, convert back to GPU tensor.

Phase 2 (future GPU optimisation):
    Replace with pre-computed torch.nn.functional.conv2d kernel.
"""

from typing import Optional

import torch
import numpy as np
from ..base import Masker, PhysicalMask, ProcessorOutput, MaskerConfig
from . import register_masker
try:
    from skimage.filters import gaussian as _gaussian_blur
except ImportError:
    _gaussian_blur = None

@register_masker("vision_blur")
class VisionBlurMasker(Masker):
    """Image occlusion via Gaussian blur in masked regions.

    Uses ``skimage.filters.gaussian`` on CPU for the blur.

    Contracts:
        - Only ``pixel_values`` is modified; ``input_ids`` and
          ``attention_mask`` pass through unchanged.
        - Originals are never mutated — the blend creates new tensors.

    Args:
        config: Optional MaskerConfig. Sigma is read from
            ``config.vision_blur.sigma`` when provided, falling back
            to the constructor default.
        sigma: Default sigma (standard deviation) for the Gaussian
            kernel. Only used when ``config`` is ``None``.
    """

    def __init__(
        self,
        config: Optional[MaskerConfig] = None,
        sigma: float = 3.0,
    ):
        super().__init__(config)
        if _gaussian_blur is None:
            raise ImportError(
                "VisionBlurMasker requires scikit-image. "
                "Install with: pip install scikit-image"
            )

        # Resolve sigma: typed config overrides constructor default
        if config is not None:
            self._sigma = config.vision_blur.sigma
        else:
            self._sigma = sigma

    # ─── Public API ───────────────────────────────────────────────────────

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """Apply Gaussian blur to masked regions of ``pixel_values``.

        For each (batch, channel) slice:
            1. Blur the entire image with ``skimage.filters.gaussian``
               using the configured sigma.
            2. Blend: keep original pixels where mask=1, use blurred
               pixels where mask=0.

        The full-image blur per channel is simpler and faster than
        per-region blurring, and produces correct results because the
        mask selects which parts of the blurred image to use.

        Args:
            processor_output: Original model inputs. Only ``pixel_values``
                is consumed; ``input_ids`` and ``attention_mask`` pass
                through unchanged.
            physical_mask: Must contain ``image_binary_mask`` of shape
                ``(N, C, H, W)`` with dtype float (1=keep, 0=blur).

        Returns:
            ProcessorOutput with blurred ``pixel_values``.
        """
        if physical_mask.image_binary_mask is None:
            return processor_output

        pixel_values = processor_output.pixel_values          # (N, C, H, W)
        mask = physical_mask.image_binary_mask                # (N, C, H, W)
        device = pixel_values.device

        # ── CPU-based per-channel Gaussian blur ──────────────────────────
        # skimage operates on numpy arrays, so we transfer to CPU once.
        im_np = pixel_values.detach().cpu().numpy()            # (N, C, H, W)
        mask_np = mask.detach().cpu().numpy()                  # (N, C, H, W)

        sigma = self._sigma
        n_batch, n_chan = im_np.shape[:2]

        blurred = np.empty_like(im_np)
        for b in range(n_batch):
            for c in range(n_chan):
                # skimage.filters.gaussian applies a 2D Gaussian to
                # (H, W) single-channel arrays
                blurred[b, c] = _gaussian_blur(
                    im_np[b, c], sigma=sigma
                )

        # Blend: original where mask=1, blurred where mask=0
        blended_np = im_np * mask_np + blurred * (1.0 - mask_np)

        # Convert back to original device and dtype
        pixel_values = torch.from_numpy(blended_np).to(
            device=device, dtype=processor_output.pixel_values.dtype
        )

        return ProcessorOutput(
            pixel_values=pixel_values,
            input_ids=processor_output.input_ids,          # pass-through
            attention_mask=processor_output.attention_mask, # pass-through
            model_type=processor_output.model_type,
        )
