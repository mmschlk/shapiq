"""Baseline masking for models of any array backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from shapiq.games.maskers._base import (
    BackendArray,
    Masker,
    coalition_masks_like,
    require_shared_backend,
)

if TYPE_CHECKING:
    from shapiq.coalitions import CoalitionArray


@dataclass(frozen=True)
class BaselineMasker[InputT: BackendArray](Masker[InputT]):
    """Masker replacing absent players' inputs with baseline values.

    Players are the trailing input axis — for tabular inputs, the columns.
    Present players keep the explanation target's input values; absent
    players are replaced by the baseline. Leading input axes become the
    explanation target shape, so a batch of instances is explained at once.
    The masked inputs carry the coalition sample axis before the player
    axis, so a model consuming ``(..., n_players)`` inputs evaluates every
    coalition in one batched call.

    The masker computes in the backend its arrays come from — NumPy, JAX,
    torch, or any Array API compatible library — and the masked inputs stay
    in that backend on the inputs' device, so the model receives exactly
    what it was trained on.

    Example:
        >>> masker = BaselineMasker(inputs=features[0], baseline=features.mean(axis=0))
        >>> predictor = ModelMaskedPredictor(masker=masker, model=model.predict)
    """

    inputs: InputT
    baseline: InputT

    def __post_init__(self) -> None:
        """Derive player and target metadata from the inputs."""
        if self.inputs.ndim < 1:
            msg = "inputs must carry at least the trailing player axis"
            raise ValueError(msg)
        require_shared_backend(self.inputs, baseline=self.baseline)
        n_players = int(self.inputs.shape[-1])
        if tuple(self.baseline.shape) != (n_players,):
            msg = (
                f"baseline must have shape ({n_players},) to pair with the "
                f"inputs' player axis, got {tuple(self.baseline.shape)}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "n_players", n_players)
        object.__setattr__(self, "target_shape", tuple(self.inputs.shape[:-1]))

    def _mask(self, coalitions: CoalitionArray) -> InputT:
        """Return model-native inputs with absent players set to the baseline."""
        masks = coalition_masks_like(coalitions, self.inputs)
        xp = array_namespace(self.inputs)
        return xp.where(masks, xp.expand_dims(self.inputs, axis=-2), self.baseline)
