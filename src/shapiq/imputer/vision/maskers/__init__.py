"""
Segmenter registry — maps string names to Masker classes.

To add a new masker, decorate the class with ``@register_masker("name")``.
The factory will resolve it via config.strategy.
"""

from ..base import Masker

_MASKER_REGISTRY: dict[str, type[Masker]] = {}


def register_masker(name: str):
    """Decorator: register a Masker class under a string key."""
    def _register(cls):
        key = name.lower()
        if key not in _MASKER_REGISTRY:
            _MASKER_REGISTRY[key] = cls
        return cls
    return _register


def get_masker(name: str) -> type[Masker]:
    """Look up a Masker class by name."""
    if name is None:
        raise TypeError("masker name must not be None")
    key = name.lower()
    if key not in _MASKER_REGISTRY:
        raise KeyError(
            f"Unknown masker '{name}'. "
            f"Registered: {list(_MASKER_REGISTRY.keys())}"
        )
    return _MASKER_REGISTRY[key]


# Built-in maskers
from .vision_mean import VisionMeanMasker
from .vision_blur import VisionBlurMasker
from .text_attention import TextAttentionMasker
from .crossmodal_mean import CrossModalMeanMasker
from .crossmodal_blur import CrossModalBlurMasker
