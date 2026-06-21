"""Vision-based explanation methods for image models."""

try:
    from .architecture import CNNArchitecture, ModelArchitectureStrategy, TransformerArchitecture
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
except ImportError as _e:

    class ImageExplainer:
        """Placeholder raised when the optional ``vision`` extra is not installed."""

        _import_error = _e

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Raise an informative ImportError pointing to the missing extra."""
            raise self._import_error


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
