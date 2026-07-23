"""Segmenter registry — maps string names to Segmenter classes.

To add a new segmenter, decorate the class with ``@register_segmenter("name")``.
The factory will resolve it via config.strategy.
"""

from collections.abc import Callable

from .base import Segmenter

_SEGMENTER_REGISTRY: dict[str, type[Segmenter]] = {}


def register_segmenter(name: str) -> Callable[[type[Segmenter]], type[Segmenter]]:
    """Decorator: register a Segmenter class under a string key."""

    def _register(cls: type[Segmenter]) -> type[Segmenter]:
        key = name.lower()
        if key not in _SEGMENTER_REGISTRY:
            _SEGMENTER_REGISTRY[key] = cls
        return cls

    return _register


def get_segmenter(name: str) -> type[Segmenter]:
    """Look up a Segmenter class by name."""
    if name is None:
        msg = "segmenter name must not be None"
        raise TypeError(msg)
    key = name.lower()
    if key not in _SEGMENTER_REGISTRY:
        msg = f"Unknown segmenter '{name}'. Registered: {list(_SEGMENTER_REGISTRY.keys())}"
        raise KeyError(msg)
    return _SEGMENTER_REGISTRY[key]


__all__ = [
    "CustomSegmenter",
    "PatchSegmenter",
    "SLICSegmenter",
    "Segmenter",
    "SegmenterConfig",
    "get_segmenter",
    "register_segmenter",
]


# Built-in segmenters — imported for registration side effects
from .custom import CustomSegmenter  # noqa: E402
from .patch import PatchSegmenter  # noqa: E402
from .slic import SLICSegmenter  # noqa: E402
