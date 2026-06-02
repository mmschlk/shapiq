"""Vision-based explanation methods for image models."""

from .architecture import ModelArchitectureStrategy, ResNetArchitecture, ViTArchitecture
from .imputer import ImageImputer
from .masking import (
    BoolMaskedPosStrategy,
    LatentMaskingStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    PixelMaskingStrategy,
    ZeroMasking,
)
from .players import (
    LatentPlayerStrategy,
    PatchStrategy,
    PixelPlayerStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
)

__all__ = [
    # Architecture
    "ModelArchitectureStrategy",
    "ResNetArchitecture",
    "ViTArchitecture",
    # Imputer
    "ImageImputer",
    # Masking
    "PixelMaskingStrategy",
    "LatentMaskingStrategy",
    "MeanColorMasking",
    "ZeroMasking",
    "BoolMaskedPosStrategy",
    "MaskTokenStrategy",
    # Players
    "PlayerStrategy",
    "PixelPlayerStrategy",
    "LatentPlayerStrategy",
    "SuperpixelStrategy",
    "PatchStrategy",
]


def __getattr__(name: str) -> object:
    if name == "ImageExplainer":
        from .explainer import ImageExplainer

        return ImageExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
