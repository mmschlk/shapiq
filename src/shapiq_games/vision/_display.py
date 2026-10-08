"""Ways to look at an image game whose players are regions of an image."""

from __future__ import annotations

import numpy as np
from PIL import Image

from shapiq.interaction_values import InteractionValues

__all__ = ["DISPLAY_GRAY", "RegionPlots", "gray_masked_image"]

DISPLAY_GRAY = 128
"""The gray that shows a region a model removes internally (e.g. by dropping its tokens)."""


class RegionPlots:
    """Heatmaps and player patches for games with :attr:`image` and :attr:`regions`."""

    image: np.ndarray
    regions: np.ndarray

    def attribution_map(self, values: InteractionValues | np.ndarray) -> np.ndarray:
        """Spread one value per player over the player's pixels, e.g. for a heatmap.

        Args:
            values: Interaction values (their first-order values are used) or one value per
                player.

        Returns:
            The value of every pixel, of shape ``(height, width)``.
        """
        if isinstance(values, InteractionValues):
            values = values.get_n_order_values(1)
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != self._n_regions():
            msg = f"Expected {self._n_regions()} values, one per player, got {values.shape[0]}."
            raise ValueError(msg)
        return values[self.regions]

    def player_images(self, *, fade: float = 0.75) -> list[Image.Image]:
        """Return one image per player: its bounding box, with other pixels faded to white.

        The images can be passed to :func:`shapiq.plot.si_graph_plot` as
        ``feature_image_patches``.

        Args:
            fade: How strongly pixels outside the player's region are faded, from ``0`` (not at
                all) to ``1`` (white). Defaults to ``0.75``.

        Returns:
            The images, in player order.
        """
        images = []
        for player in range(self._n_regions()):
            mask = self.regions == player
            rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
            box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
            crop = self.image[box].astype(float)
            outside = ~mask[box]
            crop[outside] = crop[outside] * (1.0 - fade) + 255.0 * fade
            images.append(Image.fromarray(crop.round().astype(np.uint8)))
        return images

    def _n_regions(self) -> int:
        return int(self.regions.max()) + 1


def gray_masked_image(image: np.ndarray, regions: np.ndarray, coalition: np.ndarray) -> np.ndarray:
    """Return ``image`` with the regions of the players outside ``coalition`` in gray."""
    shown = image.copy()
    shown[~np.asarray(coalition, dtype=bool).reshape(-1)[regions]] = DISPLAY_GRAY
    return shown
