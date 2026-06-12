"""Vision-based explanation methods for image models."""

from .architecture import CNNArchitecture, ModelArchitectureStrategy, TransformerArchitecture
from .imputer import ImageImputer
from .masking import (
    BoolMaskedPosStrategy,
    CNNMaskingStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    TransformerMaskingStrategy,
    ZeroMasking,
)
from .players import (
    CNNPlayerStrategy,
    PatchStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
    TransformerPlayerStrategy,
)

__all__ = [
    # Architecture
    "ModelArchitectureStrategy",
    "CNNArchitecture",
    "TransformerArchitecture",
    # Imputer
    "ImageImputer",
    # Masking
    "CNNMaskingStrategy",
    "TransformerMaskingStrategy",
    "MeanColorMasking",
    "ZeroMasking",
    "BoolMaskedPosStrategy",
    "MaskTokenStrategy",
    # Players
    "PlayerStrategy",
    "CNNPlayerStrategy",
    "TransformerPlayerStrategy",
    "SuperpixelStrategy",
    "PatchStrategy",
]


def __getattr__(name: str) -> object:
    if name == "ImageExplainer":
        from .explainer import ImageExplainer

        return ImageExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
