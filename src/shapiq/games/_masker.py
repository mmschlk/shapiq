from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray


class Masker[ModelInputT](ABC):
    """Base abstraction for converting coalitions to model-native inputs."""

    n_players: int
    target_shape: Shape

    def __call__(self, coalitions: CoalitionArray) -> ModelInputT:
        """Create masked model-native inputs for coalitions."""
        self._validate_coalitions(coalitions)
        return self._mask(coalitions)

    def _validate_coalitions(self, coalitions: CoalitionArray) -> None:
        """Validate coalition compatibility at the masker boundary."""
        if coalitions.n_players != self.n_players:
            msg = "coalitions use a different number of players"
            raise ValueError(msg)

    @abstractmethod
    def _mask(self, coalitions: CoalitionArray) -> ModelInputT:
        """Create masked inputs after base validation."""
