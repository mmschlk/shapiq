"""
shapiq.imputer.vision — Vision-language model imputation sub-package.

Provides pluggable Segmenter and Masker abstractions for explaining
vision-language model predictions via Shapley interactions.

Core pipeline::

    from shapiq.imputer.vision import VisionImputerFactory, VisionLanguageGame

    factory = VisionImputerFactory()
    imputer = factory.build(model, processor, image, text)
    game = VisionLanguageGame(imputer, batch_size=64)
    # game can now be used with any shapiq approximator
"""

from .base import (
    Segmenter,
    Masker,
    SegmenterConfig,
    MaskerConfig,
    PatchParams,
    SlicParams,
    GradientGuidedParams,
    CrossModalMeanParams,
    CrossModalBlurParams,
    VisionMeanParams,
    VisionBlurParams,
    TextAttentionParams,
    SpatialLayout,
    PhysicalMask,
    ProcessorOutput,
)
from .imputer import VisionImputer
from .factory import VisionImputerFactory
from .game import VisionLanguageGame
from .segmenters import PatchSegmenter
from .maskers import (
    VisionMeanMasker,
    VisionBlurMasker,
    TextAttentionMasker,
    CrossModalMeanMasker,
    CrossModalBlurMasker,
)

__all__ = [
    "VisionImputer",
    "VisionImputerFactory",
    "VisionLanguageGame",
    "Segmenter",
    "Masker",
    "SegmenterConfig",
    "MaskerConfig",
    "PatchParams",
    "SlicParams",
    "GradientGuidedParams",
    "CrossModalMeanParams",
    "CrossModalBlurParams",
    "VisionMeanParams",
    "VisionBlurParams",
    "TextAttentionParams",
    "SpatialLayout",
    "PhysicalMask",
    "ProcessorOutput",
    "PatchSegmenter",
    "VisionMeanMasker",
    "VisionBlurMasker",
    "TextAttentionMasker",
    "CrossModalMeanMasker",
    "CrossModalBlurMasker",
]
