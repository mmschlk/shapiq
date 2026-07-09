"""Torch game adapters."""

from __future__ import annotations

from shapiq.games.torch._callable import TorchCallableGame
from shapiq.games.torch._convert import to_jax
from shapiq.games.torch._image import ImageGame
from shapiq.games.torch._masker import BaselineMasker
from shapiq.games.torch._superpixel import SuperpixelMasker, grid_labels

__all__ = [
    "BaselineMasker",
    "ImageGame",
    "SuperpixelMasker",
    "TorchCallableGame",
    "grid_labels",
    "to_jax",
]
