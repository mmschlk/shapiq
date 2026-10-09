"""The base of the image games whose players are regions of an image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from shapiq.game import Game
from shapiq.interaction_values import InteractionValues
from shapiq_games._base import as_bool_coalitions

from ._fill import FILLS, GRAY, filled_images

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import Fill, MaskStrategy

__all__ = ["RegionGame", "resolve_removal"]


def resolve_removal(
    mask_strategy: MaskStrategy | None,
    fill: Fill | np.ndarray | None,
    *,
    default: MaskStrategy | None,
) -> MaskStrategy | None:
    """Check how a game removes its players, before any model is loaded.

    Args:
        mask_strategy: ``"mask"`` or ``"remove"`` in token space, or ``None``.
        fill: A named fill or a fill image for image space, or ``None``.
        default: The strategy when neither is given.

    Returns:
        The token-space strategy, or ``None`` for image space (a ``fill`` is given).

    Raises:
        ValueError: If both are given, or either is unknown. A fill image's shape is checked
            later, against the image the model sees.
    """
    if mask_strategy is not None and fill is not None:
        msg = "Choose mask_strategy (token space) or fill (image space), not both."
        raise ValueError(msg)
    if mask_strategy is not None and mask_strategy not in ("mask", "remove"):
        msg = f"mask_strategy must be 'mask' or 'remove', got {mask_strategy!r}."
        raise ValueError(msg)
    if isinstance(fill, str) and fill not in FILLS:
        msg = f"fill must be one of {', '.join(map(repr, FILLS))} or an image, got {fill!r}."
        raise ValueError(msg)
    if fill is not None:
        return None
    return mask_strategy or default


class RegionGame(Game):
    """Base of the image games whose players are regions of an image.

    A subclass sets :attr:`image`, :attr:`regions`, and ``_fill`` (the fill image when players are
    removed in image space, ``None`` when its model removes them itself, in token space),
    implements ``_evaluate`` for non-empty coalitions, and calls ``super().__init__`` last. The
    empty coalition is always answered with the stored :attr:`empty_value`, so a value does not
    depend on the batch it is evaluated in.

    Attributes:
        image: The image the model sees.
        regions: The player of every pixel, of shape ``(height, width)``, numbered from ``0``.
        empty_value: The value of the empty coalition before centering.
        original_model_output: The value of the grand coalition (the full image) before centering.
    """

    image: np.ndarray
    regions: np.ndarray
    _fill: np.ndarray | None

    def __init__(self, *, normalize: bool, verbose: bool) -> None:
        """Evaluate the empty and the full image and set up the game.

        Args:
            normalize: Whether to center the game such that the value of the empty coalition is
                zero.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        n_players = int(self.regions.max()) + 1
        self.empty_value = float(self._evaluate(np.zeros((1, n_players), dtype=bool))[0])
        self.original_model_output = float(self._evaluate(np.ones((1, n_players), dtype=bool))[0])
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def _evaluate(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the model output for each coalition."""
        raise NotImplementedError

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the model output with only each coalition's regions present."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.full(coalitions.shape[0], self.empty_value)
        present = coalitions.any(axis=1)  # the empty coalition is exactly the stored value
        if present.any():
            values[present] = self._evaluate(coalitions[present])
        return values

    def masked_image(self, coalition: ArrayLike) -> np.ndarray:
        """Return the image with the players outside ``coalition`` removed.

        In image space this is the image the model sees (up to rounding, for models that fill
        normalized pixels). In token space the model removes the players inside the network, so
        their regions are shown in gray.

        Args:
            coalition: The players, as a boolean or 0/1 vector of length ``n_players``.

        Returns:
            The image as a ``uint8`` array of the image's shape.
        """
        present = as_bool_coalitions(coalition)
        if self._fill is not None:
            return filled_images(self.image, self.regions, self._fill, present)[0]
        shown = self.image.copy()
        shown[~present[0][self.regions]] = GRAY
        return shown

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
        if values.shape[0] != self.n_players:
            msg = f"Expected {self.n_players} values, one per player, got {values.shape[0]}."
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
        for player in range(self.n_players):
            mask = self.regions == player
            rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
            box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
            crop = self.image[box].astype(float)
            outside = ~mask[box]
            crop[outside] = crop[outside] * (1.0 - fade) + 255.0 * fade
            images.append(Image.fromarray(crop.round().astype(np.uint8)))
        return images
