from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shapiq.games._base import Game, LinkFunction

if TYPE_CHECKING:
    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray
    from shapiq.games._masked_predictor import MaskedPredictor


@dataclass(frozen=True)
class MaskedGame[PredictionT, ValueT](Game[ValueT]):
    """Game composed from a masked predictor and a link function."""

    masked_predictor: MaskedPredictor[PredictionT]
    link_function: LinkFunction[PredictionT, ValueT]

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
