"""Vision-based explanation methods for image models."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_LAZY_MODULES: dict[str, str] = {
    "ModelArchitectureStrategy": "architecture",
    "CNNArchitecture": "architecture",
    "TransformerArchitecture": "architecture",
    "ImageExplainer": "explainer",
    "ImageImputer": "imputer",
    "CNNMaskingStrategy": "masking",
    "TransformerMaskingStrategy": "masking",
    "MeanColorMasking": "masking",
    "ZeroMasking": "masking",
    "BoolMaskedPosStrategy": "masking",
    "MaskTokenStrategy": "masking",
    "PlayerStrategy": "players",
    "CNNPlayerStrategy": "players",
    "TransformerPlayerStrategy": "players",
    "SuperpixelStrategy": "players",
    "PatchStrategy": "players",
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
        CNNArchitecture,
        ModelArchitectureStrategy,
        TransformerArchitecture,
    )
    from .explainer import ImageExplainer
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
    # Explainer
    "ImageExplainer",
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
