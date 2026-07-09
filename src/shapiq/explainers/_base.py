from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shapiq.games import Game
from shapiq.interactions import InteractionIndex

if TYPE_CHECKING:
    from shapiq.explanations import ExplanationArray

_SHIPPED_SYMBOLS = {
    "SV": "SV",
    "BV": "BV",
    "SII": "SII",
    "BII": "BII",
    "CHII": "CHII",
    "STII": "STII",
    "k-SII": "KSII",
    "FSII": "FSII",
    "FBII": "FBII",
    "kADD-SHAP": "KADDSHAP",
    "SGV": "SGV",
    "BGV": "BGV",
    "CHGV": "CHGV",
    "IGV": "IGV",
    "EGV": "EGV",
    "JointSV": "JointSV",
    "Moebius": "Moebius",
    "Co-Moebius": "CoMoebius",
}

_INDEX_MEMBERS = (
    "name",
    "order_semantics",
    "includes_empty_interaction",
    "preserves_value",
    "generalizes",
    "resolve_order",
)


def missing_index_members(index: object) -> list[str]:
    """Return the ``InteractionIndex`` protocol members absent from the index."""
    return [member for member in _INDEX_MEMBERS if not hasattr(index, member)]


def reject_common_index_mistakes(index: object) -> None:
    """Raise teaching errors for strings and classes passed as indices."""
    if isinstance(index, str):
        symbol = _SHIPPED_SYMBOLS.get(index, "SII")
        msg = (
            f"interaction indices are singleton values: pass shapiq.{symbol} itself "
            f"(the order, if any, goes to the explainer), not the string {index!r}"
        )
        raise TypeError(msg)
    if isinstance(index, type):
        msg = (
            f"interaction indices are singleton values, not classes: pass the "
            f"{index.__name__} value itself — for a custom index pass an instance"
        )
        raise TypeError(msg)


class Explainer[ValueT, GameT: Game](ABC):
    """Base abstraction for objects that explain games."""

    game: GameT
    index: InteractionIndex
    order: int

    def __init__(
        self,
        game: GameT,
        index: InteractionIndex,
        *,
        order: int | None = None,
    ) -> None:
        """Initialize shared explainer metadata.

        Args:
            game: Game to explain.
            index: The interaction index to compute, passed as the singleton
                value itself, such as ``shapiq.SII``.
            order: Maximum interaction order of the explanation. Probabilistic
                values (SV, BV) fix it at one, transforms (Moebius,
                Co-Moebius) default to all orders, and every other index
                requires an explicit order.
        """
        reject_common_index_mistakes(index)
        if not isinstance(index, InteractionIndex):
            missing = missing_index_members(index)
            hint = f" (missing protocol members: {', '.join(missing)})" if missing else ""
            msg = (
                "index must be an interaction index value such as shapiq.SII, "
                f"got {type(index).__name__}{hint}"
            )
            raise TypeError(msg)
        self.game = game
        self.index = index
        self.order = index.resolve_order(order, n_players=game.n_players)

    @property
    def interaction_index(self) -> str:
        """Return the name of the explained interaction index."""
        return self.index.name

    def __call__(self) -> ExplanationArray[ValueT]:
        """Alias explain()."""
        return self.explain()

    @abstractmethod
    def explain(self) -> ExplanationArray[ValueT]:
        """Return explanations for the bound game."""
