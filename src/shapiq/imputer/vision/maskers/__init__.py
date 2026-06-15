"""Masker registry — maps string names to Masker classes.

To add a new masker, decorate the class with ``@register_masker("name")``.
The factory will resolve it via config.strategy.
"""

from collections.abc import Callable

from .base import Masker

_MASKER_REGISTRY: dict[str, type[Masker]] = {}


def register_masker(name: str) -> Callable[[type[Masker]], type[Masker]]:
    """Decorator: register a Masker class under a string key."""

    def _register(cls: type[Masker]) -> type[Masker]:
        key = name.lower()
        if key not in _MASKER_REGISTRY:
            _MASKER_REGISTRY[key] = cls
        return cls

    return _register


def get_masker(name: str) -> type[Masker]:
    """Look up a Masker class by name."""
    if name is None:
        msg = "masker name must not be None"
        raise TypeError(msg)
    key = name.lower()
    if key not in _MASKER_REGISTRY:
        msg = f"Unknown masker '{name}'. Registered: {list(_MASKER_REGISTRY.keys())}"
        raise KeyError(msg)
    return _MASKER_REGISTRY[key]


__all__ = [
    "CrossModalBlurMasker",
    "CrossModalMeanMasker",
    "TextAttentionMasker",
    "VisionBlurMasker",
    "VisionMeanMasker",
    "Masker",
    "MaskerConfig",
    "get_masker",
    "register_masker",
]


# Built-in maskers — imported for registration side effects
from .crossmodal_blur import CrossModalBlurMasker  # noqa: E402
from .crossmodal_mean import CrossModalMeanMasker  # noqa: E402
from .text_attention import TextAttentionMasker  # noqa: E402
from .vision_blur import VisionBlurMasker  # noqa: E402
from .vision_mean import VisionMeanMasker  # noqa: E402
