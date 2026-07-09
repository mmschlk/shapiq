"""Chunked coalition evaluation for torch models."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import jax.numpy as jnp
import torch

from shapiq._shape import validate_int
from shapiq.games._masked_predictor import MaskedPredictor

if TYPE_CHECKING:
    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray
    from shapiq.games._base import Model
    from shapiq.games._masker import Masker


@dataclass(frozen=True)
class ChunkedMaskedPredictor(MaskedPredictor[torch.Tensor]):
    """Masked predictor streaming coalitions through a torch model in chunks.

    Models with big per-instance inputs (images) need flat
    ``(batch, *instance)`` tensors, and materializing masked inputs for
    every coalition at once is the memory hazard of explaining them. This
    predictor owns the efficient use of its masker: coalitions stream
    through the masker and model in chunks whose flat instance count stays
    within ``batch_size`` (explanation-target batches divide the coalition
    samples per chunk; at least one coalition is masked per chunk, and the
    final chunk carries the remainder), so only one chunk of masked inputs
    is alive at a time per device. The trailing ``instance_axes`` input
    axes form one model instance — ``(channels, height, width)`` for
    images, the player axis for tabular models — and are flattened around
    the model call, so the model receives batches exactly as trained,
    including one empty batch when a coalition array holds no coalitions.

    Devices follow the tensors users already have: each masked chunk moves
    to the model's parameter device when that differs from the masker's,
    so the masker's tensors stay put and only chunk-sized traffic crosses
    devices; ``device`` overrides the inference for models without
    parameters or with pinned placements. The device is resolved on every
    prediction, so moving the model after construction is picked up.
    Compose with ``MaskedGame`` and a link function for the full game.

    Example:
        >>> masker = SuperpixelMasker(inputs=image, baseline=0.0, labels=labels)
        >>> predictor = ChunkedMaskedPredictor(masker=masker, model=cnn, batch_size=64)
        >>> game = MaskedGame(masked_predictor=predictor, link_function=to_jax)
    """

    masker: Masker[torch.Tensor]
    model: Model[torch.Tensor, torch.Tensor]
    batch_size: int = 64
    instance_axes: int = 3
    no_grad: bool = True
    device: torch.device | str | None = None

    def __post_init__(self) -> None:
        """Validate the evaluation policy."""
        validate_int("batch_size", self.batch_size, minimum=1)
        validate_int("instance_axes", self.instance_axes, minimum=1)
        if self.device is not None:
            object.__setattr__(self, "device", torch.device(self.device))

    @property
    def n_players(self) -> int:
        """Return the fixed number of players."""
        return self.masker.n_players

    @property
    def target_shape(self) -> Shape:
        """Return the explanation target shape."""
        return self.masker.target_shape

    def _predict(self, coalitions: CoalitionArray) -> torch.Tensor:
        """Stream coalition chunks through the masker and model."""
        if self.no_grad:
            with torch.no_grad():
                return self._predict_chunked(coalitions)
        return self._predict_chunked(coalitions)

    def _predict_chunked(self, coalitions: CoalitionArray) -> torch.Tensor:
        device = torch.device(self.device) if self.device is not None else _model_device(self.model)
        if coalitions.shape == ():
            return self._forward(self.masker(coalitions), device)
        n_samples = coalitions.shape[-1]
        leading = (slice(None),) * (len(coalitions.shape) - 1)
        instances_per_sample = prod(
            jnp.broadcast_shapes(self.target_shape, coalitions.shape[:-1]),
        )
        samples_per_chunk = max(1, self.batch_size // max(instances_per_sample, 1))
        chunks: list[torch.Tensor] = []
        sample_axis = 0
        for start in range(0, max(n_samples, 1), samples_per_chunk):
            stop = min(start + samples_per_chunk, n_samples)
            masked = self.masker(coalitions[(*leading, slice(start, stop))])
            sample_axis = masked.ndim - self.instance_axes - 1
            chunks.append(self._forward(masked, device))
            # release the chunk before masking the next one: only one chunk
            # of masked inputs is alive at a time per device
            del masked
        return torch.cat(chunks, dim=sample_axis) if len(chunks) > 1 else chunks[0]

    def _forward(self, masked: torch.Tensor, device: torch.device | None) -> torch.Tensor:
        """Run one flat instance batch through the model, keeping leading axes."""
        if device is not None and masked.device != device:
            masked = masked.to(device)
        flat = masked.reshape(-1, *masked.shape[-self.instance_axes :])
        predictions = self.model(flat)
        return predictions.reshape(*masked.shape[: -self.instance_axes], *predictions.shape[1:])


def _model_device(model: object) -> torch.device | None:
    """Return the device of a model's first parameter, if it has any."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    first = next(iter(parameters()), None)
    return first.device if isinstance(first, torch.Tensor) else None
