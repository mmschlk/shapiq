"""Torch models and outputs: calls on the module's device, values from tensors."""

from itertools import chain
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Float

from shapiq.masker.base import require_eager
from shapiq.value.value import JaxValue

from .base import _opaque_model_spec, as_value, infer_model_spec, predict

_FLOAT_DTYPES = {
    torch.float16: jnp.dtype(jnp.float16),
    torch.bfloat16: jnp.dtype(jnp.bfloat16),
    torch.float32: jnp.dtype(jnp.float32),
    torch.float64: jnp.dtype(jnp.float64),
}


@infer_model_spec.register(torch.nn.Module)
def _(model: torch.nn.Module, x_sample: object = None) -> jax.ShapeDtypeStruct:  # noqa: ARG001
    """Estimate output dtype from the first floating parameter or buffer."""
    for value in chain(model.parameters(), model.buffers()):
        dtype = _FLOAT_DTYPES.get(value.dtype)
        if dtype is not None:
            return jax.ShapeDtypeStruct((), jax.dtypes.canonicalize_dtype(dtype))
    return _opaque_model_spec()


@as_value.register(torch.Tensor)
def _(outputs: torch.Tensor) -> Float[JaxValue, " n_samples"]:
    """Return the tensor's values, detached and moved to the CPU."""
    if outputs.dtype == torch.bfloat16:
        return as_value(jax.dlpack.from_dlpack(outputs.detach().cpu().contiguous()))
    return as_value(outputs.numpy(force=True))


@predict.register(torch.nn.Module)
def _(model: torch.nn.Module, x: Any) -> Float[JaxValue, " n_samples"]:  # noqa: ANN401
    """Return the module's outputs on the module's device, without gradients."""
    require_eager("torch modules", x)
    leaf = next(chain(model.parameters(), model.buffers()), None)
    inputs = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    with torch.no_grad():
        return as_value(model(inputs if leaf is None else inputs.to(leaf.device)))
