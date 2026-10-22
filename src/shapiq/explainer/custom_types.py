"""Types for the shapiq explainer module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from shapiq.game import Game
    from shapiq.interaction_values import InteractionValues

ExplainerIndices = Literal[
    "SV", "SII", "k-SII", "STII", "FSII", "BV", "BII", "FBII", "Moebius", "Fourier"
]
# the explainers estimating with approximators have no route to the Moebius and Fourier transforms
ValidApproximationExplainerIndices = Literal[
    "SV", "SII", "k-SII", "STII", "FSII", "BV", "BII", "FBII"
]
ValidNNExplainerIndices = Literal["SV"]
ValidProductKernelExplainerIndices = Literal["SV"]


class ApproximatorLike(Protocol):
    """What the explainers need from their ``approximator``.

    Every :class:`~shapiq.approximator.Approximator` satisfies it, and so does the exact
    computation of ``approximator="exact"``, which is not an approximator.
    """

    @property
    def index(self) -> str:
        """The interaction index to compute."""
        ...

    @property
    def max_order(self) -> int:
        """The maximum interaction order to compute."""
        ...

    def __call__(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray]
    ) -> InteractionValues:
        """Computes the interaction values of the game with at most ``budget`` evaluations."""
        ...

    def set_random_state(self, random_state: int | None = None) -> None:
        """Sets the random state."""
        ...
