from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.dlpack
import jax.numpy as jnp
import torch

from shapiq.games import CallableGame

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq._shape import ShapeLike
    from shapiq.coalitions import CoalitionArray


class TorchCallableGame[ValueT](CallableGame[ValueT]):
    """Callable game with default torch/JAX boundary conversion."""

    no_grad: bool
    detach: bool

    def __init__(
        self,
        fn: Callable[[object], object],
        n_players: int,
        target_shape: ShapeLike = (),
        *,
        coalition_converter: Callable[[CoalitionArray], object] | None = None,
        value_converter: Callable[[object], ValueT] | None = None,
        value_shape: ShapeLike = (),
        no_grad: bool = True,
        detach: bool = True,
    ) -> None:
        """Initialize a torch-backed callable game."""
        object.__setattr__(self, "no_grad", no_grad)
        object.__setattr__(self, "detach", detach)
        super().__init__(
            fn=fn,
            n_players=n_players,
            target_shape=target_shape,
            coalition_converter=coalition_converter or _coalitions_to_torch,
            value_converter=value_converter or self._torch_to_jax,
            value_shape=value_shape,
        )

    def _call(self, coalitions: CoalitionArray) -> ValueT:
        """Evaluate under torch.no_grad when configured."""
        if self.no_grad:
            with torch.no_grad():
                return super()._call(coalitions)
        return super()._call(coalitions)

    def _torch_to_jax(self, value: object) -> ValueT:
        """Convert torch outputs to JAX arrays when possible."""
        if isinstance(value, torch.Tensor):
            tensor = value.detach() if self.detach else value
            return cast("ValueT", jax.dlpack.from_dlpack(tensor))
        return cast("ValueT", jnp.asarray(value))


def _coalitions_to_torch(coalitions: CoalitionArray) -> torch.Tensor:
    """Convert coalitions to torch bool tensors, using DLPack when possible."""
    dense = coalitions.to_dense()
    if hasattr(dense, "__dlpack__"):
        return torch.from_dlpack(dense).to(dtype=torch.bool)
    return torch.as_tensor(dense, dtype=torch.bool)
