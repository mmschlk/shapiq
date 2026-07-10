from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shapiq.games import Game
from shapiq.interactions import InteractionIndex, validate_interaction_metadata

if TYPE_CHECKING:
    from shapiq.explanations import ExplanationArray


_SHIPPED_EXAMPLES = {
    "SV": "SV()",
    "BV": "BV()",
    "WeightedBV": "WeightedBV(p=0.5)",
    "SII": "SII(order=2)",
    "BII": "BII(order=2)",
    "WeightedBII": "WeightedBII(p=0.5, order=2)",
    "CHII": "CHII(order=2)",
    "STII": "STII(order=2)",
    "k-SII": "KSII(order=2)",
    "FSII": "FSII(order=2)",
    "FBII": "FBII(order=2)",
    "WeightedFBII": "WeightedFBII(p=0.5, order=2)",
    "kADD-SHAP": "KADDSHAP(order=2)",
    "SGV": "SGV(order=2)",
    "BGV": "BGV(order=2)",
    "CHGV": "CHGV(order=2)",
    "IGV": "IGV(order=2)",
    "EGV": "EGV(order=2)",
    "JointSV": "JointSV(order=2)",
    "Moebius": "Moebius()",
    "Co-Moebius": "CoMoebius()",
}

_INDEX_MEMBERS = (
    "name",
    "order",
    "order_semantics",
    "includes_empty_interaction",
    "preserves_value",
    "generalizes",
)


def missing_index_members(index: object) -> list[str]:
    """Return the ``InteractionIndex`` protocol members absent from the index."""
    return [member for member in _INDEX_MEMBERS if not hasattr(index, member)]


def reject_common_index_mistakes(index: object) -> None:
    """Raise teaching errors for strings and index classes passed as indices."""
    if isinstance(index, str):
        example = _SHIPPED_EXAMPLES.get(index, "SII(order=2)")
        msg = f"interaction indices are objects: pass shapiq.{example} instead of {index!r}"
        raise TypeError(msg)
    if isinstance(index, type):
        msg = (
            f"pass an index instance such as {index.__name__}(order=2), "
            f"not the {index.__name__} class"
        )
        raise TypeError(msg)


class Explainer[ValueT, GameT: Game](ABC):
    """Base abstraction for objects that explain games."""

    game: GameT
    index: InteractionIndex

    def __init__(self, game: GameT, index: InteractionIndex) -> None:
        """Initialize shared explainer metadata."""
        reject_common_index_mistakes(index)
        if not isinstance(index, InteractionIndex):
            missing = missing_index_members(index)
            hint = f" (missing protocol members: {', '.join(missing)})" if missing else ""
            msg = (
                "index must be an interaction index object such as shapiq.SII(order=2), "
                f"got {type(index).__name__}{hint}"
            )
            raise TypeError(msg)
        order = game.n_players if index.order is None else index.order
        validate_interaction_metadata(
            index_name=index.name,
            order=order,
            orientation="undirected",
            n_players=game.n_players,
        )
        self.game = game
        self.index = index

    @property
    def interaction_index(self) -> str:
        """Return the name of the explained interaction index."""
        return self.index.name

    @property
    def order(self) -> int:
        """Return the maximum interaction order of the explanation.

        Indices with order ``None`` (the Moebius and Co-Moebius transforms)
        resolve to the full number of players.
        """
        order = self.index.order
        return self.game.n_players if order is None else order

    def __call__(self) -> ExplanationArray[ValueT]:
        """Alias explain()."""
        return self.explain()

    @abstractmethod
    def explain(self) -> ExplanationArray[ValueT]:
        """Return explanations for the bound game."""
