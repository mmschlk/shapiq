"""Dispatched conversion of model-native predictions to game values."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flextype import flexdispatch
from jax import Array

from shapiq._lazy_types import TORCH_TENSOR


@flexdispatch
def to_values(predictions: object) -> Array:
    """Convert model-native predictions to JAX-backed game values.

    The fallback handles everything ``jnp.asarray`` accepts: JAX arrays,
    NumPy arrays, Python numbers, and nested sequences. Backend-specific
    conversions register lazily on the prediction's type — passing a torch
    tensor materializes the torch handler (DLPack import with a
    host-memory fallback) without shapiq ever importing torch on its own.
    ``MaskedGame`` uses this conversion when no link function is passed,
    and custom link functions typically end with it.

    Args:
        predictions: Model-native predictions.

    Returns:
        The predictions as a JAX array in the game's value space.
    """
    return jnp.asarray(predictions)


@to_values.delayed_register(TORCH_TENSOR)
def _register_torch_values(_: type) -> None:
    """Materialize the torch conversion the first time a tensor arrives."""
    import shapiq.games.torch._convert  # noqa: F401, PLC0415 - registers the handler


def to_host_array(array: object, dtype: type[np.generic] | None = None) -> np.ndarray:
    """Return any array-API input as a host NumPy array.

    Exact host-side computations (tree routing, label validation) start
    here. NumPy views most backends directly; inputs it refuses
    (device-resident tensors) come home through ``to_values``.

    Args:
        array: Any array-like — NumPy, JAX, torch tensors, nested sequences.
        dtype: The NumPy dtype of the returned array, or ``None`` to keep
            the input's dtype.

    Returns:
        The input as a host NumPy array.
    """
    try:
        host = np.asarray(array, dtype=dtype)
    except TypeError:
        host = np.asarray(to_values(array), dtype=dtype)
    return host
