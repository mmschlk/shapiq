from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, cast

from shapiq._shape import ShapeLike, normalize_shape, validate_n_players
from shapiq.games._base import Game

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.coalitions import CoalitionArray


@dataclass(frozen=True)
class CallableGame[ValueT](Game[ValueT]):
    """Game adapter for callables that already behave like games.

    The callable receives coalitions (converted by ``coalition_converter``
    when one is given) and its outputs become game values (converted by
    ``value_converter`` when one is given). No framework call policy is
    applied; for torch callables use ``shapiq.games.torch.TorchCallableGame``,
    which fills both converters and evaluates without autograd.
    """

    fn: Callable[[object], object]
    n_players: int
    target_shape: ShapeLike = ()
    _: KW_ONLY
    coalition_converter: Callable[[CoalitionArray], object] | None = None
    value_converter: Callable[[object], ValueT] | None = None
    value_shape: ShapeLike = ()

    def __post_init__(self) -> None:
        """Normalize metadata."""
        object.__setattr__(self, "n_players", validate_n_players(self.n_players))
        object.__setattr__(self, "target_shape", normalize_shape(self.target_shape))
        object.__setattr__(self, "value_shape", normalize_shape(self.value_shape))

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
