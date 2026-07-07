"""Baseline masking for torch models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from shapiq.games._masker import Masker
from shapiq.games.torch._callable import _coalitions_to_torch

if TYPE_CHECKING:
    from shapiq.coalitions import CoalitionArray


@dataclass(frozen=True)
class BaselineMasker(Masker[torch.Tensor]):
    """Masker replacing absent players' inputs with baseline values.

    Players are the trailing input axis — for tabular inputs, the columns.
    Present players keep the explanation target's input values; absent
    players are replaced by the baseline. Leading input axes become the
    explanation target shape, so a batch of instances is explained at once.
    The masked inputs carry the coalition sample axis before the player
    axis, so a model consuming ``(..., n_players)`` inputs evaluates every
    coalition in one batched call.

    Example:
        >>> masker = BaselineMasker(inputs=x, baseline=x_train.mean(dim=0))
        >>> predictor = ModelMaskedPredictor(masker=masker, model=model)
    """

    inputs: torch.Tensor
    baseline: torch.Tensor

    def __post_init__(self) -> None:
        """Derive player and target metadata from the inputs."""
        if self.inputs.ndim < 1:
            msg = "inputs must carry at least the trailing player axis"
            raise ValueError(msg)
        n_players = int(self.inputs.shape[-1])
        if self.baseline.shape != (n_players,):
            msg = (
                f"baseline must have shape ({n_players},) to pair with the "
                f"inputs' player axis, got {tuple(self.baseline.shape)}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "n_players", n_players)
        object.__setattr__(self, "target_shape", tuple(self.inputs.shape[:-1]))

    def _mask(self, coalitions: CoalitionArray) -> torch.Tensor:
        """Return model-native inputs with absent players set to the baseline."""
        masks = _coalitions_to_torch(coalitions)
        return torch.where(masks, self.inputs.unsqueeze(-2), self.baseline)
