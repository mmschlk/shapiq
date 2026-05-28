from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shapiq.games import Game
from shapiq.interactions import (
    InteractionIndexName,
    InteractionOrientation,
    validate_interaction_metadata,
)

if TYPE_CHECKING:
    from shapiq.explanations import ExplanationArray


class Explainer[ValueT, GameT: Game](ABC):
    """Base abstraction for objects that explain games."""

    game: GameT
    interaction_index: InteractionIndexName
    order: int
    orientation: InteractionOrientation

    def __init__(
        self,
        game: GameT,
        interaction_index: InteractionIndexName,
        order: int,
        orientation: InteractionOrientation = "undirected",
    ) -> None:
        """Initialize shared explainer metadata."""
        validate_interaction_metadata(
            interaction_index=interaction_index,
            order=order,
            orientation=orientation,
            n_players=game.n_players,
        )
        self.game = game
        self.interaction_index = interaction_index
        self.order = order
        self.orientation = orientation

    def __call__(self) -> ExplanationArray[ValueT]:
        """Alias explain()."""
        return self.explain()

    @abstractmethod
    def explain(self) -> ExplanationArray[ValueT]:
        """Return explanations for the bound game."""
