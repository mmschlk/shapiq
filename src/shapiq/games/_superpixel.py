"""Superpixel masking for image models of any array backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from array_api_compat import array_namespace, device

from shapiq._shape import broadcast_shapes, validate_int
from shapiq.games._masker import (
    BackendArray,
    Masker,
    coalition_masks_like,
    require_shared_backend,
)
from shapiq.games._values import to_host_array

if TYPE_CHECKING:
    from shapiq.coalitions import CoalitionArray


@dataclass(frozen=True)
class SuperpixelMasker[InputT: BackendArray](Masker[InputT]):
    """Masker replacing absent superpixels' pixels with baseline values.

    Players are superpixels: ``labels`` assigns every pixel of the trailing
    ``(height, width)`` axes to one player, and the superpixel ids must
    cover ``0 .. n_players - 1``. Present superpixels keep the explanation
    target's pixel values; absent superpixels are replaced by the baseline,
    which broadcasts against the image (a plain float for a gray value,
    per-channel values shaped ``(channels, 1, 1)``, or a full baseline
    image). Inputs are channel-first images with trailing
    ``(channels, height, width)`` axes; leading axes become the explanation
    target shape. The masked images carry the coalition sample axis before
    the image axes.

    The masker computes in the backend the inputs come from — NumPy, JAX,
    torch, or any Array API compatible library — and masked images stay on
    the inputs' device. Labels are index metadata and may come from any
    backend (``grid_labels`` returns a host NumPy map); they are converted
    once at construction. The masker only expands coalition masks to pixel
    masks and applies one ``where``; streaming masked images through a
    model efficiently is the predictor's job (see
    ``ChunkedMaskedPredictor`` for torch models).

    Example:
        >>> labels = grid_labels(height=27, width=27, grid=(3, 3))
        >>> masker = SuperpixelMasker(inputs=image, baseline=0.0, labels=labels)
    """

    inputs: InputT
    baseline: InputT | float
    labels: Any

    def __post_init__(self) -> None:
        """Derive player and target metadata from the image and label map."""
        xp = array_namespace(self.inputs)
        if self.inputs.ndim < 3:
            msg = (
                "inputs must be channel-first images with trailing "
                "(channels, height, width) axes; add a channel axis for "
                "single-channel images"
            )
            raise ValueError(msg)
        image_shape = tuple(self.inputs.shape[-3:])
        raw_labels = to_host_array(self.labels)
        if np.issubdtype(raw_labels.dtype, np.floating):
            msg = (
                "labels must hold non-negative integer superpixel ids; cast "
                "float label maps to integers first"
            )
            raise ValueError(msg)
        if raw_labels.ndim != 2 or raw_labels.shape != image_shape[-2:]:
            hint = ""
            if raw_labels.shape == tuple(self.inputs.shape[-3:-1]):
                hint = (
                    "; the labels match the leading image axes, so the image is "
                    "likely channels-last (height, width, channels) - permute it "
                    "to channel-first"
                )
            msg = (
                f"labels must assign one superpixel per pixel with shape "
                f"{image_shape[-2:]}, got {raw_labels.shape}{hint}"
            )
            raise ValueError(msg)
        if int(raw_labels.min()) < 0:
            msg = "labels must hold non-negative integer superpixel ids"
            raise ValueError(msg)
        n_players = int(raw_labels.max()) + 1
        if int(np.unique(raw_labels).size) != n_players:
            msg = (
                f"superpixel ids must cover 0 .. {n_players - 1} with no gaps, "
                f"but only {int(np.unique(raw_labels).size)} distinct ids appear; "
                "players without pixels cannot influence the game - renumber "
                "the map to consecutive ids"
            )
            raise ValueError(msg)
        try:
            baseline_namespace = array_namespace(self.baseline)
        except TypeError:
            baseline_array = xp.asarray(
                self.baseline, dtype=self.inputs.dtype, device=device(self.inputs)
            )
        else:
            del baseline_namespace
            require_shared_backend(self.inputs, baseline=self.baseline)
            baseline_array = cast("InputT", self.baseline)
        try:
            broadcast_shapes(tuple(baseline_array.shape), image_shape)
        except ValueError as error:
            msg = (
                f"baseline with shape {tuple(baseline_array.shape)} does not "
                f"broadcast against the image shape {image_shape}; pass a "
                "float, per-channel values shaped (channels, 1, 1), or a "
                "baseline image"
            )
            raise ValueError(msg) from error
        object.__setattr__(self, "baseline", baseline_array)
        object.__setattr__(
            self,
            "labels",
            xp.asarray(raw_labels.astype(np.int64), device=device(self.inputs)),
        )
        object.__setattr__(self, "n_players", n_players)
        object.__setattr__(self, "target_shape", tuple(self.inputs.shape[:-3]))

    def _mask(self, coalitions: CoalitionArray) -> InputT:
        """Return masked images with absent superpixels set to the baseline."""
        masks = cast("BackendArray", coalition_masks_like(coalitions, self.inputs))
        xp = array_namespace(self.inputs)
        pixel_masks = xp.take(masks, xp.reshape(self.labels, (-1,)), axis=-1)
        pixel_masks = xp.reshape(pixel_masks, (*tuple(masks.shape)[:-1], *self.labels.shape))
        return xp.where(
            xp.expand_dims(pixel_masks, axis=-3),
            xp.expand_dims(self.inputs, axis=-4),
            self.baseline,
        )


def grid_labels(height: int, width: int, grid: tuple[int, int] = (3, 3)) -> np.ndarray:
    """Return a superpixel label map partitioning an image into a grid.

    Pixels are bucketed into ``grid`` roughly equal bands per axis (exact
    when the image dimensions are divisible), and superpixels are numbered
    row-major, so a ``(3, 3)`` grid yields nine players with player ``0``
    in the top-left corner. The map is host NumPy; maskers convert it to
    their inputs' backend at construction.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        grid: Number of superpixel ``(rows, columns)``.

    Returns:
        An integer ``(height, width)`` array of superpixel ids.

    Raises:
        ValueError: If the grid does not fit the image.
    """
    rows, columns = grid
    validate_int("height", height, minimum=1)
    validate_int("width", width, minimum=1)
    validate_int("grid rows", rows, minimum=1)
    validate_int("grid columns", columns, minimum=1)
    if rows > height or columns > width:
        msg = (
            f"a {grid} grid does not fit a {height}x{width} image: every "
            "superpixel needs at least one pixel"
        )
        raise ValueError(msg)
    row_ids = np.arange(height) * rows // height
    column_ids = np.arange(width) * columns // width
    return row_ids[:, None] * columns + column_ids[None, :]
