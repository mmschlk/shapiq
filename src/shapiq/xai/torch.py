"""Torch models and outputs: calls on the module's device, values from tensors."""

from typing import Any

import numpy as np
import torch
from jaxtyping import Float

from shapiq.value.value import JaxValue

from .base import as_value, predict, require_eager


@as_value.register(torch.Tensor)
def _(outputs: torch.Tensor) -> Float[JaxValue, " n_samples"]:
    """Return the tensor's values, detached and moved to the CPU."""
    return as_value(outputs.numpy(force=True))


@predict.register(torch.nn.Module)
def _(model: torch.nn.Module, x: Any) -> Float[JaxValue, " n_samples"]:  # noqa: ANN401
    """Return the module's outputs on the module's device, without gradients."""
    require_eager("torch modules", x)
    device = next(model.parameters(), torch.empty(0)).device
    inputs = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    with torch.no_grad():
        return as_value(model(inputs.to(device)))
