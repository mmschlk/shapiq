"""shapiq.imputer.vision — Vision-language model imputation sub-package.

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
    CrossModalBlurParams,
    CrossModalMeanParams,
    CustomSegmenterParams,
    GradientGuidedParams,
    Masker,
    MaskerConfig,
    PatchParams,
    PhysicalMask,
    ProcessorOutput,
    Segmenter,
    SegmenterConfig,
    SlicParams,
    SpatialLayout,
    TextAttentionParams,
    VisionBlurParams,
    VisionMeanParams,
)
from .factory import VisionImputerFactory
from .game import VisionLanguageGame
from .imputer import VisionImputer
from .maskers import (
    CrossModalBlurMasker,
    CrossModalMeanMasker,
    TextAttentionMasker,
    VisionBlurMasker,
    VisionMeanMasker,
)
from .segmenters import CustomSegmenter, PatchSegmenter

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
    "CustomSegmenter",
    "CustomSegmenterParams",
    "VisionMeanMasker",
    "VisionBlurMasker",
    "TextAttentionMasker",
    "CrossModalMeanMasker",
    "CrossModalBlurMasker",
]
