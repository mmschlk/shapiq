from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from shapiq._shape import ShapeLike, normalize_shape, validate_n_players
from shapiq.games._base import Game

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.coalitions import CoalitionArray


@dataclass(frozen=True)
class CallableGame[ValueT](Game[ValueT]):
    """Game adapter for callables that already behave like games."""

    fn: Callable[[object], object]
    n_players: int
    target_shape: ShapeLike = ()
    coalition_converter: Callable[[CoalitionArray], object] | None = None
    value_converter: Callable[[object], ValueT] | None = None

    def __post_init__(self) -> None:
        """Normalize metadata."""
        object.__setattr__(self, "n_players", validate_n_players(self.n_players))
        object.__setattr__(self, "target_shape", normalize_shape(self.target_shape))

    def _call(self, coalitions: CoalitionArray) -> ValueT:
        """Evaluate the wrapped callable."""
        native_coalitions = (
            self.coalition_converter(coalitions)
            if self.coalition_converter is not None
            else coalitions
        )
        raw_values = self.fn(native_coalitions)
        if self.value_converter is not None:
            return self.value_converter(raw_values)
        return cast("ValueT", raw_values)
