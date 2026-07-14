from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from shapiq.games import CallableGame
from shapiq.games.torch._convert import to_jax

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq._shape import ShapeLike
    from shapiq.coalitions import CoalitionArray


class TorchCallableGame[ValueT](CallableGame[ValueT]):
    """Callable game owning the torch call policy for coalition callables.

    The torch call policy lives in ``shapiq.games.torch``, owned by one
    adapter per entry style: this game wraps callables that consume
    coalitions and return torch tensors, and ``ChunkedMaskedPredictor``
    owns the masked path (masker + model). Wrapped callables receive
    coalitions as boolean torch tensors, run without building autograd
    graphs, and their outputs re-enter the stack as JAX arrays through the
    DLPack boundary — so a plain ``fn(coalitions) -> torch.Tensor`` behaves
    like any other game without torch ceremony at the call site.

    Example:
        >>> game = TorchCallableGame(fn=coalition_scorer, n_players=8)
        >>> explanation = Regression(game, SV()).sample(64).explain()
    """

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
        """Initialize a torch-backed callable game.

        Args:
            fn: Callable evaluating coalitions to model-native values.
            n_players: The fixed number of players.
            target_shape: Explanation-target axes of the game.
            coalition_converter: Conversion applied to coalitions before the
                callable; defaults to boolean torch tensors via DLPack.
            value_converter: Conversion applied to the callable's outputs;
                defaults to JAX arrays via DLPack with a host-memory
                fallback.
            value_shape: Trailing value axes of one evaluation.
            no_grad: Whether to evaluate under ``torch.no_grad()`` so no
                autograd graph is built.
            detach: Whether the default value conversion detaches outputs
                from an existing autograd graph first.
        """
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
        return cast("ValueT", to_jax(value, detach=self.detach))


def _coalitions_to_torch(coalitions: CoalitionArray) -> torch.Tensor:
    """Convert coalitions to torch bool tensors, keeping the DLPack device.

    DLPack import preserves the coalitions' device, so a GPU JAX backend
    hands the wrapped callable GPU-resident masks with no copy; inputs
    torch cannot import come through host memory instead.
    """
    dense = coalitions.to_dense()
    try:
        return torch.from_dlpack(dense).to(dtype=torch.bool)
    except (AttributeError, BufferError, RuntimeError, TypeError, ValueError):
        return torch.as_tensor(np.asarray(dense).copy(), dtype=torch.bool)
