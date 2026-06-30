"""CrossModalBlurMasker — composite masker with Gaussian-blur image occlusion.

Follows the Composite Pattern: internally instantiates two atomic maskers
(VisionBlurMasker + TextAttentionMasker) and delegates image/text occlusion
to each respectively. Owns no low-level tensor math itself.

This is the counterpart of ``CrossModalMeanMasker``; the only difference is
the image-side occlusion strategy (Gaussian blur instead of zero-out mean).

Registered as ``"crossmodal_blur"`` in the masker registry.

.. note::
    Requires ``VisionBlurMasker``. See ``VisionBlurMasker`` for
    the CPU Gaussian blur implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq.imputer.vision.base import PhysicalMask, ProcessorOutput
from shapiq.imputer.vision.maskers.base import Masker, MaskerConfig, VisionBlurParams

from . import register_masker
from .text_attention import TextAttentionMasker
from .vision_blur import VisionBlurMasker


@register_masker("crossmodal_blur")
class CrossModalBlurMasker(Masker):
    """Cross-modal occlusion orchestrator with Gaussian-blur image occlusion.

    Delegates:
        - Image occlusion → VisionBlurMasker (registered as ``"vision_blur"``)
        - Text occlusion  → TextAttentionMasker (registered as ``"text_attn"``)

    The composite itself performs no element-wise operations.

    Usage:
        From a notebook or experiment::

            from ImputerFactory import MaskerConfig
            cfg = MaskerConfig(strategy="crossmodal_blur")
            imputer = factory.build(model, processor, img, txt, masker_config=cfg)

    Registered as ``"crossmodal_blur"``.
    """

    def __init__(self, config: MaskerConfig | None = None) -> None:
        """Initialize the cross-modal blur masker."""
        super().__init__(config)
        cfg = config or MaskerConfig()
        sigma = cfg.params.sigma if hasattr(cfg.params, "sigma") else 3.0
        vision_cfg = MaskerConfig(
            strategy="vision_blur",
            params=VisionBlurParams(sigma=sigma),
        )
        self._vision_masker = VisionBlurMasker(config=vision_cfg)
        self._text_masker = TextAttentionMasker(config=config)

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """Apply blur image occlusion + attention-mask text occlusion.

        Args:
            processor_output: Original model inputs.
            physical_mask: Contains both ``image_binary_mask`` (for blur)
                and ``text_attention_mask`` (for attention swapping).

        Returns:
            ProcessorOutput with:
                - ``pixel_values``: Gaussian-blurred in masked regions.
                - ``attention_mask``: Replaced by text coalition mask.
                - ``input_ids``: Unchanged.
        """
        # 1. Delegate image occlusion to VisionBlurMasker (blur + blend)
        masked = self._vision_masker.apply(processor_output, physical_mask)

        # 2. Delegate text occlusion to TextAttentionMasker
        #    (swaps attention_mask, passes pixel_values through)
        return self._text_masker.apply(masked, physical_mask)
