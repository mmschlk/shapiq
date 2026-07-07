from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shapiq._shape import ShapeLike, normalize_shape
from shapiq.games._base import Game, LinkFunction

if TYPE_CHECKING:
    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray
    from shapiq.games._masked_predictor import MaskedPredictor


@dataclass(frozen=True)
class MaskedGame[PredictionT, ValueT](Game[ValueT]):
    """Game composed from a masked predictor and a link function.

    The game owns the value-space declaration: ``value_shape`` states the
    internal shape of the values the link function produces per coalition,
    with the default declaring scalar values.
    """

    masked_predictor: MaskedPredictor[PredictionT]
    link_function: LinkFunction[PredictionT, ValueT]
    value_shape: ShapeLike = ()

    def __post_init__(self) -> None:
        """Normalize metadata."""
        object.__setattr__(self, "value_shape", normalize_shape(self.value_shape))

    @property
    def n_players(self) -> int:
        """Return the fixed number of players."""
        return self.masked_predictor.n_players

    @property
    def target_shape(self) -> Shape:
        """Return the explanation target shape."""
        return self.masked_predictor.target_shape

    def _call(self, coalitions: CoalitionArray) -> ValueT:
        """Compose masked predictor and link function."""
        return self.link_function(self.masked_predictor(coalitions))
