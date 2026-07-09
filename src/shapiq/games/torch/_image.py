"""Chunked torch image game."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import jax.numpy as jnp
import torch

from shapiq._shape import ShapeLike, normalize_shape, validate_int
from shapiq.games._base import Game, LinkFunction, Model

if TYPE_CHECKING:
    from jax import Array

    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray
    from shapiq.games._masker import Masker

_IMAGE_AXES = 3


@dataclass(frozen=True)
class ImageGame(Game["Array"]):
    """Game streaming masked images through a torch model in chunks.

    Image models need flat ``(batch, channels, height, width)`` inputs, and
    materializing masked images for every coalition at once is the memory
    hazard of image explanation. The game therefore owns the efficient use
    of its masker: coalitions stream through the masker and model in chunks
    whose flat image count stays within ``batch_size`` (explanation-target
    batches divide the coalition samples per chunk; at least one coalition
    is masked per chunk, and the final chunk carries the remainder), so
    only one chunk of masked images is alive at a time. The masker itself
    stays a plain mask-and-replace component (``SuperpixelMasker``), the
    model consumes flat image batches as trained — including one empty
    batch when a coalition array holds no coalitions — and the link
    function maps the collected predictions to game values.

    Example:
        >>> labels = grid_labels(height=27, width=27, grid=(3, 3))
        >>> masker = SuperpixelMasker(inputs=image, baseline=0.0, labels=labels)
        >>> game = ImageGame(masker=masker, model=cnn, link_function=to_jax, batch_size=64)
        >>> explanation = ExactExplainer(game, SV()).explain()
    """

    masker: Masker[torch.Tensor]
    model: Model[torch.Tensor, torch.Tensor]
    link_function: LinkFunction[torch.Tensor, Array]
    batch_size: int = 64
    value_shape: ShapeLike = ()
    no_grad: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate the evaluation policy."""
        validate_int("batch_size", self.batch_size, minimum=1)
        object.__setattr__(self, "value_shape", normalize_shape(self.value_shape))

    @property
    def n_players(self) -> int:
        """Return the fixed number of players."""
        return self.masker.n_players

    @property
    def target_shape(self) -> Shape:
        """Return the explanation target shape."""
        return self.masker.target_shape

    def _call(self, coalitions: CoalitionArray) -> Array:
        """Stream coalition chunks through the masker and model, then link."""
        if self.no_grad:
            with torch.no_grad():
                return self.link_function(self._predict_chunked(coalitions))
        return self.link_function(self._predict_chunked(coalitions))

    def _predict_chunked(self, coalitions: CoalitionArray) -> torch.Tensor:
        if coalitions.shape == ():
            return self._forward(self.masker(coalitions))
        n_samples = coalitions.shape[-1]
        leading = (slice(None),) * (len(coalitions.shape) - 1)
        images_per_sample = prod(jnp.broadcast_shapes(self.target_shape, coalitions.shape[:-1]))
        samples_per_chunk = max(1, self.batch_size // max(images_per_sample, 1))
        chunks: list[torch.Tensor] = []
        sample_axis = 0
        for start in range(0, max(n_samples, 1), samples_per_chunk):
            stop = min(start + samples_per_chunk, n_samples)
            masked = self.masker(coalitions[(*leading, slice(start, stop))])
            sample_axis = masked.ndim - _IMAGE_AXES - 1
            chunks.append(self._forward(masked))
            # release the chunk before masking the next one: only one chunk
            # of masked images is alive at a time
            del masked
        return torch.cat(chunks, dim=sample_axis) if len(chunks) > 1 else chunks[0]

    def _forward(self, masked: torch.Tensor) -> torch.Tensor:
        """Run one flat image batch through the model, keeping leading axes."""
        flat = masked.reshape(-1, *masked.shape[-_IMAGE_AXES:])
        predictions = self.model(flat)
        return predictions.reshape(*masked.shape[:-_IMAGE_AXES], *predictions.shape[1:])
