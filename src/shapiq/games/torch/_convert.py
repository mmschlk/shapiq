"""Torch-to-JAX value conversion for link functions."""

from __future__ import annotations

from typing import cast

import jax.dlpack
import jax.numpy as jnp
import torch
from jax import Array

from shapiq.games._values import to_values


def to_jax(values: object, *, detach: bool = True) -> Array:
    """Convert torch outputs to JAX arrays, using DLPack when possible.

    Link functions that compose torch predictions into game values end with
    this conversion; non-tensor inputs fall back to ``jnp.asarray``.

    Args:
        values: Torch tensor or array-like predictions.
        detach: Whether to detach tensors from the autograd graph first.

    Returns:
        The values as a JAX array.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.detach() if detach else values
        # DLPack import requires compact striding; slices and transposed
        # views coming out of link functions rarely have it
        tensor = tensor.contiguous()
        try:
            return cast("Array", jax.dlpack.from_dlpack(tensor))
        except (RuntimeError, TypeError, ValueError):
            # JAX cannot import tensors living on devices it has no backend
            # for (CUDA/MPS torch with CPU JAX); copy through host memory
            return jnp.asarray(tensor.cpu().numpy())
    return jnp.asarray(values)


@to_values.register(torch.Tensor)
def _tensor_to_values(predictions: torch.Tensor) -> Array:
    """Convert torch predictions through the DLPack boundary."""
    return to_jax(predictions)
