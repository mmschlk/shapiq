"""Superpixel masking for torch image models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from shapiq._shape import validate_int
from shapiq.games._masker import Masker
from shapiq.games.torch._callable import _coalitions_to_torch

if TYPE_CHECKING:
    from shapiq.coalitions import CoalitionArray


@dataclass(frozen=True)
class SuperpixelMasker(Masker[torch.Tensor]):
    """Masker replacing absent superpixels' pixels with baseline values.

    Players are superpixels: ``labels`` assigns every pixel of the trailing
    ``(height, width)`` axes to one player, and the superpixel ids must
    cover ``0 .. n_players - 1``. Present superpixels keep the explanation
    target's pixel values; absent superpixels are replaced by the baseline,
    which broadcasts against the image (a plain float or scalar tensor for
    a gray value, per-channel values shaped ``(channels, 1, 1)``, or a full
    baseline image). Inputs are channel-first images with trailing
    ``(channels, height, width)`` axes; leading axes become the explanation
    target shape. The masked images carry the coalition sample axis before
    the image axes and live on the inputs' device. The masker only expands
    coalition masks to pixel masks and applies one ``where``; batching
    masked images through a model efficiently is the game's job (see
    ``ImageGame``).

    Example:
        >>> labels = grid_labels(height=27, width=27, grid=(3, 3))
        >>> masker = SuperpixelMasker(inputs=image, baseline=0.0, labels=labels)
    """

    inputs: torch.Tensor
    baseline: torch.Tensor | float
    labels: torch.Tensor

    def __post_init__(self) -> None:
        """Derive player and target metadata from the image and label map."""
        if not isinstance(self.baseline, torch.Tensor):
            object.__setattr__(
                self,
                "baseline",
                torch.as_tensor(self.baseline, dtype=self.inputs.dtype, device=self.inputs.device),
            )
        baseline = cast("torch.Tensor", self.baseline)
        if self.inputs.ndim < 3:
            msg = (
                "inputs must be channel-first images with trailing "
                "(channels, height, width) axes; add a channel axis for "
                "single-channel images"
            )
            raise ValueError(msg)
        image_shape = tuple(self.inputs.shape[-3:])
        if self.labels.ndim != 2 or tuple(self.labels.shape) != image_shape[-2:]:
            hint = ""
            if tuple(self.labels.shape) == tuple(self.inputs.shape[-3:-1]):
                hint = (
                    "; the labels match the leading image axes, so the image is "
                    "likely channels-last (height, width, channels) - permute it "
                    "to channel-first"
                )
            msg = (
                f"labels must assign one superpixel per pixel with shape "
                f"{image_shape[-2:]}, got {tuple(self.labels.shape)}{hint}"
            )
            raise ValueError(msg)
        if self.labels.dtype.is_floating_point or bool(self.labels.min() < 0):
            msg = (
                "labels must hold non-negative integer superpixel ids; cast "
                "float label maps with labels.long()"
            )
            raise ValueError(msg)
        n_players = int(self.labels.max()) + 1
        present = torch.unique(self.labels)
        if int(present.numel()) != n_players:
            msg = (
                f"superpixel ids must cover 0 .. {n_players - 1} with no gaps, "
                f"but only {int(present.numel())} distinct ids appear; players "
                "without pixels cannot influence the game - renumber densely "
                "with torch.unique(labels, return_inverse=True)[1].reshape(labels.shape)"
            )
            raise ValueError(msg)
        for name, tensor in (("baseline", baseline), ("labels", self.labels)):
            if tensor.device != self.inputs.device:
                msg = (
                    f"{name} lives on {tensor.device} but the inputs live on "
                    f"{self.inputs.device}; keep the masker's tensors on one device"
                )
                raise ValueError(msg)
        try:
            torch.broadcast_shapes(tuple(baseline.shape), image_shape)
        except RuntimeError as error:
            msg = (
                f"baseline with shape {tuple(baseline.shape)} does not "
                f"broadcast against the image shape {image_shape}; pass a "
                "float, per-channel values shaped (channels, 1, 1), or a "
                "baseline image"
            )
            raise ValueError(msg) from error
        object.__setattr__(self, "n_players", n_players)
        object.__setattr__(self, "target_shape", tuple(self.inputs.shape[:-3]))

    def _mask(self, coalitions: CoalitionArray) -> torch.Tensor:
        """Return masked images with absent superpixels set to the baseline."""
        masks = _coalitions_to_torch(coalitions).to(self.inputs.device)
        pixel_masks = masks[..., self.labels]
        return torch.where(
            pixel_masks.unsqueeze(-3),
            self.inputs.unsqueeze(-4),
            self.baseline,
        )


def grid_labels(height: int, width: int, grid: tuple[int, int] = (3, 3)) -> torch.Tensor:
    """Return a superpixel label map partitioning an image into a grid.

    Pixels are bucketed into ``grid`` roughly equal bands per axis (exact
    when the image dimensions are divisible), and superpixels are numbered
    row-major, so a ``(3, 3)`` grid yields nine players with player ``0``
    in the top-left corner.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        grid: Number of superpixel ``(rows, columns)``.

    Returns:
        An integer ``(height, width)`` tensor of superpixel ids.

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
    row_ids = torch.arange(height) * rows // height
    column_ids = torch.arange(width) * columns // width
    return row_ids[:, None] * columns + column_ids[None, :]
