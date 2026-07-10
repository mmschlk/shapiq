from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray
    from shapiq.games._base import Model
    from shapiq.games._masker import Masker


class MaskedPredictor[PredictionT](ABC):
    """Base abstraction for coalition-aware prediction."""

    n_players: int
    target_shape: Shape

    def __call__(self, coalitions: CoalitionArray) -> PredictionT:
        """Predict for coalitions."""
        self._validate_coalitions(coalitions)
        return self._predict(coalitions)

    def _validate_coalitions(self, coalitions: CoalitionArray) -> None:
        """Validate coalition compatibility at the predictor boundary."""
        if coalitions.n_players != self.n_players:
            msg = "coalitions use a different number of players"
            raise ValueError(msg)

    @abstractmethod
    def _predict(self, coalitions: CoalitionArray) -> PredictionT:
        """Predict after base validation."""


@dataclass(frozen=True)
class ModelMaskedPredictor[ModelInputT, PredictionT](MaskedPredictor[PredictionT]):
    """Masked predictor formed by composing a masker and callable model.

    The composition is backend-generic and applies no framework call
    policy: a torch model called here builds autograd graphs during the
    forward pass. Torch models belong in
    ``shapiq.games.torch.ChunkedMaskedPredictor``, which owns the torch
    policy for the masked path (no-grad evaluation, device placement, and
    chunked streaming).
    """

    masker: Masker[ModelInputT]
    model: Model[ModelInputT, PredictionT]

    @property
    def n_players(self) -> int:
        """Return the fixed number of players."""
        return self.masker.n_players

    @property
    def target_shape(self) -> Shape:
        """Return the explanation target shape."""
        return self.masker.target_shape

    def _predict(self, coalitions: CoalitionArray) -> PredictionT:
        """Compose masker and model."""
        return self.model(self.masker(coalitions))
