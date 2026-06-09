"""
VisionMeanMasker — Pure image occlusion via multiplicative binary mask.

Operates exclusively on pixel_values. Must never touch input_ids or
attention_mask.
"""

import torch

from ..base import Masker, PhysicalMask, ProcessorOutput
from . import register_masker


@register_masker("vision_mean")
class VisionMeanMasker(Masker):
    """
    Pure image occlusion via multiplicative binary mask.

    Since CLIP/SigLIP inputs are normalized (mean ≈ 0), zeroing out
    pixels is equivalent to filling with the dataset mean.

    Registered as ``"vision_mean"``.
    """

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        pixel_values = processor_output.pixel_values
        if physical_mask.image_binary_mask is not None:
            pixel_values = pixel_values * physical_mask.image_binary_mask.to(pixel_values.device)
        return ProcessorOutput(
            pixel_values=pixel_values,
            input_ids=processor_output.input_ids,
            attention_mask=processor_output.attention_mask,
            model_type=processor_output.model_type,
        )
