"""Exact computation of the interaction values for the explainers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, get_args

from shapiq.approximator.base import ValidApproximationIndices
from shapiq.game_theory.exact import ExactComputer

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from shapiq.game import Game
    from shapiq.interaction_values import InteractionValues


class ExactSolver:
    """Computes the exact interaction values for ``approximator="exact"`` of the explainers.

    It evaluates the game on all ``2**n`` coalitions with
    :class:`~shapiq.game_theory.exact.ExactComputer`. Unlike an approximator, it ignores the
    budget and is deterministic.
    """

    def __init__(self, n: int, max_order: int, index: ValidApproximationIndices) -> None:
        """Initialize the ExactSolver.

        Args:
            n: The number of players.

            max_order: The maximum interaction order to compute.

            index: The interaction index to compute.

        Raises:
            ValueError: If the index is not one of the approximation indices.
        """
        valid_indices = get_args(ValidApproximationIndices)
        if index not in valid_indices:
            msg = f"Invalid index '{index}'. Valid indices are: {valid_indices}."
            raise ValueError(msg)
        if n > 16:
            warnings.warn(
                f"The number of players ({n}) is greater than 16. Computing the exact values "
                f"evaluates the game on all 2**{n} coalitions and may take a long time.",
                stacklevel=2,
            )
        self.n = n
        self.max_order = max_order
        self.index: ValidApproximationIndices = index

    def __call__(
        self,
        budget: int,  # noqa: ARG002
        game: Game | Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        """Computes the exact interaction values of the game.

        Args:
            budget: Not used, since all ``2**n`` coalitions are evaluated.

            game: The game to compute the interaction values of.

        Returns:
            The exact interaction values.
        """
        # ExactComputer caches the game values, and the explainers refit their imputer in place
        # for every explained point, so every call needs a fresh computer
        return ExactComputer(game, n_players=self.n)(self.index, self.max_order)

    def set_random_state(self, random_state: int | None = None) -> None:
        """Does nothing, since the exact computation is deterministic."""
