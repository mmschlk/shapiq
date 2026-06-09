"""Segmenter registry — maps string names to Segmenter classes.

To add a new segmenter, decorate the class with ``@register_segmenter("name")``.
The factory will resolve it via config.strategy.
"""

from ..base import Segmenter

_SEGMENTER_REGISTRY: dict[str, type[Segmenter]] = {}


def register_segmenter(name: str):
    """Decorator: register a Segmenter class under a string key."""

    def _register(cls):
        key = name.lower()
        if key not in _SEGMENTER_REGISTRY:
            _SEGMENTER_REGISTRY[key] = cls
        return cls

    return _register


def get_segmenter(name: str) -> type[Segmenter]:
    """Look up a Segmenter class by name."""
    if name is None:
        raise TypeError("segmenter name must not be None")
    key = name.lower()
    if key not in _SEGMENTER_REGISTRY:
        raise KeyError(
            f"Unknown segmenter '{name}'. Registered: {list(_SEGMENTER_REGISTRY.keys())}"
        )
    return _SEGMENTER_REGISTRY[key]


# Built-in segmenters
from .custom import CustomSegmenter
from .patch import PatchSegmenter
from .slic import SLICSegmenter
