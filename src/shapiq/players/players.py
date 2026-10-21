# Copyright (c) 2026 shapiq contributors
"""Base abstraction for batched sets of players."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import prod
from typing import TYPE_CHECKING, Any

from ndlike.jax_like import JaxLikeImplementation

if TYPE_CHECKING:
    from shapiq.coalition import Coalition


class Players[C: Coalition](JaxLikeImplementation[Any], ABC):
    """An abstract JAX-like batch whose elements are sets of players.

    ``shape`` describes the batch of player sets, excluding each set's internal
    structure. A scalar ``Players`` object has shape ``()`` and represents one
    player set, which can be shared across a batch of games.

    The type parameter ``C`` declares the compatible coalition representations.
    Admissibility also depends on the particular player sets: for example, a
    discrete player set may accept boolean JAX arrays whose final axis has one
    entry per player. That axis is internal to a coalition, so only the leading
    axes participate in broadcasting with ``shape``.

    Subclasses implement the representation-specific array hooks and coalition
    validation. The base class chooses neither a player nor a coalition backing.
    """

    @property
    def ndim(self) -> int:
        """Return the number of player-set batch dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the number of player sets in the batch."""
        return prod(self.shape)

    @abstractmethod
    def is_valid(
        self,
        coalition: object,
        *,
        batch_shape: tuple[int, ...] | None = None,
    ) -> bool:
        """Check whether an entire coalition batch is admissible.

        Args:
            coalition: A candidate coalition batch, including unsupported types.
            batch_shape: An optional batch to which these player sets must
                broadcast without enlarging it, such as a game's shape. If
                omitted, use this object's ``shape``. The coalition's logical
                batch shape must broadcast with this batch; it may enlarge the
                resulting evaluation batch.

        Returns:
            One Python boolean for the whole batch. Return ``False`` for an
            incompatible type, internal structure, or batch shape. Validation
            should normally inspect static metadata such as type, dtype, and
            shape rather than iterate over coalitions or inspect array values.
        """

    def __contains__(self, coalition: object) -> bool:
        """Check coalition admissibility for this batch of player sets."""
        return self.is_valid(coalition)
