"""
CrossModalMeanMasker — Composite masker for Vision-Language Models.

Composite Pattern: instantiates VisionMeanMasker + TextAttentionMasker
and delegates image/text occlusion respectively.
"""

from typing import Optional

from ..base import Masker, PhysicalMask, ProcessorOutput, MaskerConfig
from . import register_masker
from .vision_mean import VisionMeanMasker
from .text_attention import TextAttentionMasker


@register_masker("crossmodal_mean")
class CrossModalMeanMasker(Masker):
    """
    Cross-modal occlusion orchestrator for VLMs.

    Delegates:
        - Image occlusion → VisionMeanMasker (``"vision_mean"``)
        - Text occlusion  → TextAttentionMasker (``"text_attn"``)
    """

    def __init__(self, config: Optional[MaskerConfig] = None):
        super().__init__(config)
        self._vision_masker = VisionMeanMasker(config=config)
        self._text_masker = TextAttentionMasker(config=config)

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        masked = self._vision_masker.apply(processor_output, physical_mask)
        masked = self._text_masker.apply(masked, physical_mask)
        return masked
