"""Superpixel segmentation of images into players."""

from __future__ import annotations

import warnings

import numpy as np

from shapiq_games._optional import require

__all__ = ["get_superpixels"]

_MAX_RETRIES = 20


def get_superpixels(image: np.ndarray, n_segments: int) -> np.ndarray:
    """Segment an RGB image into (at most) ``n_segments`` SLIC superpixels.

    SLIC may return fewer segments than requested; the request is then increased (up to 20 times)
    until enough segments are found. Segments beyond ``n_segments`` are merged into the last one,
    so the labels are exactly ``1, ..., n_segments`` whenever enough segments exist.

    Args:
        image: The image as an array of shape ``(height, width, 3)``.
        n_segments: The requested number of superpixels.

    Returns:
        The superpixel label of every pixel, of shape ``(height, width)``, with labels starting at
        ``1``.
    """
    segmentation = require("skimage.segmentation", purpose="superpixel segmentation")
    superpixels = segmentation.slic(image, n_segments=n_segments, start_label=1, slic_zero=True)
    requested = n_segments
    for _ in range(_MAX_RETRIES):
        if np.unique(superpixels).shape[0] >= n_segments:
            break
        requested += 1
        superpixels = segmentation.slic(image, n_segments=requested, start_label=1, slic_zero=True)
    n_found = np.unique(superpixels).shape[0]
    if n_found >= n_segments:
        return np.clip(superpixels, 1, n_segments)
    warnings.warn(
        f"Found only {n_found} superpixels instead of {n_segments}.", UserWarning, stacklevel=2
    )
    # relabel to 1, ..., n_found
    _, labels = np.unique(superpixels, return_inverse=True)
    return labels.reshape(superpixels.shape) + 1
