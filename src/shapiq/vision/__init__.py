"""Vision-based explanation methods for image models."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_LAZY_MODULES: dict[str, str] = {
    "ModelArchitecture": "architecture",
    "ClassificationArchitecture": "architecture",
    "ViTClassificationArchitecture": "architecture",
    "ImageExplainer": "explainer",
    "ImageImputer": "imputer",
    "PixelBasedMaskingStrategy": "masking",
    "LatentBasedMaskingStrategy": "masking",
    "MeanColorMasking": "masking",
    "ZeroMasking": "masking",
    "BlurMasking": "masking",
    "DatasetMeanMasking": "masking",
    "MarginalSampling": "masking",
    "InpaintingMasking": "masking",
    "BoolMaskedPosStrategy": "masking",
    "MaskTokenStrategy": "masking",
    "PlayerStrategy": "players",
    "PixelBasedPlayerStrategy": "players",
    "LatentBasedPlayerStrategy": "players",
    "SuperpixelStrategy": "players",
    "PatchStrategy": "players",
    "GridStrategy": "players",
    "CustomPlayerStrategy": "players",
    "labels_to_masks": "players",
}


def _missing_extra_explainer(err: ImportError) -> type:
    """Return a stand-in ``ImageExplainer`` that raises when the vision extra is absent."""

    class ImageExplainer:
        """Placeholder raised when the optional ``vision`` extra is not installed."""

        _import_error = err

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Raise an informative ImportError pointing to the missing extra."""
            raise self._import_error

    return ImageExplainer


def __getattr__(name: str) -> object:
    """Import vision members lazily (PEP 562) so ``torch`` loads only when they are used."""
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ImportError as err:
        if name == "ImageExplainer":
            return _missing_extra_explainer(err)
        raise
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from .architecture import (
        ClassificationArchitecture,
        ModelArchitecture,
        ViTClassificationArchitecture,
    )
    from .explainer import ImageExplainer
    from .imputer import ImageImputer
    from .masking import (
        BlurMasking,
        BoolMaskedPosStrategy,
        DatasetMeanMasking,
        InpaintingMasking,
        LatentBasedMaskingStrategy,
        MarginalSampling,
        MaskTokenStrategy,
        MeanColorMasking,
        PixelBasedMaskingStrategy,
        ZeroMasking,
    )
    from .players import (
        CustomPlayerStrategy,
        GridStrategy,
        LatentBasedPlayerStrategy,
        PatchStrategy,
        PixelBasedPlayerStrategy,
        PlayerStrategy,
        SuperpixelStrategy,
        labels_to_masks,
    )


__all__ = [
    # Architecture
    "ModelArchitecture",
    "ClassificationArchitecture",
    "ViTClassificationArchitecture",
    # Explainer
    "ImageExplainer",
    # Imputer
    "ImageImputer",
    # Masking
    "PixelBasedMaskingStrategy",
    "LatentBasedMaskingStrategy",
    "MeanColorMasking",
    "ZeroMasking",
    "BlurMasking",
    "DatasetMeanMasking",
    "MarginalSampling",
    "InpaintingMasking",
    "BoolMaskedPosStrategy",
    "MaskTokenStrategy",
    # Players
    "PlayerStrategy",
    "PixelBasedPlayerStrategy",
    "LatentBasedPlayerStrategy",
    "SuperpixelStrategy",
    "PatchStrategy",
    "GridStrategy",
    "CustomPlayerStrategy",
    "labels_to_masks",
]
