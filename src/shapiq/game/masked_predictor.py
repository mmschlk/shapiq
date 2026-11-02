"""Masked predictor game for explaining model predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from shapiq.game import BoolFinitePlayerGame, FinitePlayerGame
from shapiq.value import Value

if TYPE_CHECKING:
    import jax
    from jaxtyping import Shaped, UInt8

    from shapiq.value.value import Model


@dataclass(frozen=True, slots=True, weakref_slot=True)
class MaskedPredictorGame[P, In: Value, Out: Value](FinitePlayerGame[P, Out]):
    """Masked predictor formed by composing a masker and callable model."""

    masker: FinitePlayerGame[P, In]
    model: Model[In, Out]

    @property
    def n_players(self) -> int:
        """Return the fixed number of players."""
        return self.players.n_players

    @override
    def _eval_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[Out, ...]:
        """Mask the point to explain with the baseline according to the coalition and evaluate the model."""
        return self.model(self.masker(coalition))


@dataclass(frozen=True, slots=True, weakref_slot=True)
class BaselineMasker[P, V: Value](BoolFinitePlayerGame[P, V]):
    """Masker that returns a baseline value for all coalitions."""

    point: V
    baseline: V

    @property
    def n_players(self) -> int:
        """Return the fixed number of players."""
        return self.players.n_players
