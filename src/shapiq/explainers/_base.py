from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shapiq.games import Game
from shapiq.interactions import InteractionIndex, validate_interaction_metadata

if TYPE_CHECKING:
    from shapiq.explanations import ExplanationArray
    from shapiq.interactions import InteractionIndexName, InteractionOrientation


class Explainer[ValueT, GameT: Game](ABC):
    """Base abstraction for objects that explain games."""

    game: GameT
    index: InteractionIndex

    def __init__(self, game: GameT, index: InteractionIndex) -> None:
        """Initialize shared explainer metadata."""
        if isinstance(index, str):
            msg = f"interaction indices are objects: pass shapiq.SII(order=2) instead of {index!r}"
            raise TypeError(msg)
        if not isinstance(index, InteractionIndex):
            msg = (
                "index must be an interaction index object such as shapiq.SII(order=2), "
                f"got {type(index).__name__}"
            )
            raise TypeError(msg)
        validate_interaction_metadata(
            interaction_index=index.name,
            order=index.order,
            orientation=index.orientation,
            n_players=game.n_players,
        )
        self.game = game
        self.index = index

    @property
    def interaction_index(self) -> InteractionIndexName:
        """Return the name of the explained interaction index."""
        return self.index.name

    @property
    def order(self) -> int:
        """Return the maximum interaction order of the explanation."""
        return self.index.order

    @property
    def orientation(self) -> InteractionOrientation:
        """Return the interaction orientation of the explained index."""
        return self.index.orientation

    def __call__(self) -> ExplanationArray[ValueT]:
        """Alias explain()."""
        return self.explain()

    @abstractmethod
    def explain(self) -> ExplanationArray[ValueT]:
        """Return explanations for the bound game."""
