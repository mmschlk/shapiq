"""TextAttentionMasker — Pure text occlusion via attention_mask replacement.

Operates exclusively on attention_mask. Must never touch pixel_values.
"""

from __future__ import annotations

from ..base import PhysicalMask, ProcessorOutput
from ..maskers.base import Masker
from . import register_masker


@register_masker("text_attn")
class TextAttentionMasker(Masker):
    """Pure text occlusion via attention_mask replacement.

    Registered as ``"text_attn"``.
    """

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        attention_mask = processor_output.attention_mask
        if physical_mask.text_attention_mask is not None:
            attention_mask = physical_mask.text_attention_mask.to(attention_mask.device)
        return ProcessorOutput(
            pixel_values=processor_output.pixel_values,
            input_ids=processor_output.input_ids,
            attention_mask=attention_mask,
            model_type=processor_output.model_type,
        )
