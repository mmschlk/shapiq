"""Removing regions in image space: their pixels are replaced by a fill."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

from ._display import DISPLAY_GRAY
from ._preprocess import as_rgb_array

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix
    from shapiq_games.typing import Fill

__all__ = ["fill_image", "filled_images"]


def fill_image(image: np.ndarray, fill: Fill | np.ndarray) -> np.ndarray:
    """Return the image that shows through where regions are removed.

    Args:
        image: The RGB image of shape ``(height, width, 3)``.
        fill: ``"mean"`` (the image's mean color), ``"gray"``, ``"black"``, ``"blur"`` (a blurred
            copy), or an image of the same shape.

    Returns:
        The fill as an RGB ``uint8`` image of the image's shape.

    Raises:
        ValueError: If ``fill`` is unknown or an image of another shape.
    """
    if isinstance(fill, np.ndarray):
        baseline = as_rgb_array(fill)
        if baseline.shape != image.shape:
            msg = f"A fill image must have the image's shape {image.shape}, got {baseline.shape}."
            raise ValueError(msg)
        return baseline
    if fill == "mean":
        color = image.reshape(-1, 3).mean(axis=0).round()
        return np.broadcast_to(color.astype(np.uint8), image.shape).copy()
    if fill == "gray":
        return np.full_like(image, DISPLAY_GRAY)
    if fill == "black":
        return np.zeros_like(image)
    if fill == "blur":
        radius = max(image.shape[:2]) / 32
        return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)))
    msg = f"fill must be 'mean', 'gray', 'black', 'blur', or an image, got {fill!r}."
    raise ValueError(msg)


def filled_images(
    image: np.ndarray, regions: np.ndarray, fill: np.ndarray, coalitions: CoalitionMatrix
) -> np.ndarray:
    """Return one image per coalition, with the regions of the absent players filled.

    Args:
        image: The RGB image of shape ``(height, width, 3)``.
        regions: The player of every pixel, of shape ``(height, width)``.
        fill: The fill of the image's shape (see :func:`fill_image`).
        coalitions: The boolean coalitions, of shape ``(n_coalitions, n_players)``.

    Returns:
        The images, of shape ``(n_coalitions, height, width, 3)``.
    """
    images = np.repeat(image[None], coalitions.shape[0], axis=0)
    for i, coalition in enumerate(coalitions):
        absent = ~coalition[regions]
        images[i][absent] = fill[absent]
    return images
