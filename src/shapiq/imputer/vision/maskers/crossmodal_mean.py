"""CrossModalMeanMasker — Composite masker for Vision-Language Models.

Composite Pattern: instantiates VisionMeanMasker + TextAttentionMasker
and delegates image/text occlusion respectively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq.imputer.vision.base import PhysicalMask, ProcessorOutput
from shapiq.imputer.vision.maskers.base import Masker, MaskerConfig

from . import register_masker
from .text_attention import TextAttentionMasker
from .vision_mean import VisionMeanMasker


@register_masker("crossmodal_mean")
class CrossModalMeanMasker(Masker):
    """Cross-modal occlusion orchestrator for VLMs.

    Delegates:
        - Image occlusion → VisionMeanMasker (``"vision_mean"``)
        - Text occlusion  → TextAttentionMasker (``"text_attn"``)
    """

    def __init__(self, config: MaskerConfig | None = None) -> None:
        """Initialize the cross-modal mean masker."""
        super().__init__(config)
        self._vision_masker = VisionMeanMasker(config=config)
        self._text_masker = TextAttentionMasker(config=config)

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """Apply cross-modal mean occlusion (image mean + text attention)."""
        masked = self._vision_masker.apply(processor_output, physical_mask)
        return self._text_masker.apply(masked, physical_mask)
