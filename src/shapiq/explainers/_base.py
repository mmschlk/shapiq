"""Index-binding validation shared by every explainer entry point.

There is no explainer base class: an explainer is anything that maps a
game to a game, and each entry point binds its own ``game``, ``index``,
and resolved ``order`` as frozen attributes. What is shared is the
gauntlet an index object runs before binding — the teaching errors for
strings and classes, the protocol conformance check, and the metadata
validation that resolves the explanation's maximum interaction order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.interactions import InteractionIndex, validate_interaction_metadata

if TYPE_CHECKING:
    from shapiq.games import Game

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
    "min_interaction_size",
    "includes_empty_interaction",
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


def validate_index_binding(game: Game, index: InteractionIndex) -> int:
    """Validate an index object against a game and resolve the order.

    Args:
        game: The game the explainer binds.
        index: The interaction index object the explainer estimates.

    Returns:
        The explanation's maximum interaction order: the index's own order,
        or the full player count for indices with order ``None`` (the
        Moebius and Co-Moebius transforms).

    Raises:
        TypeError: If the index is a string, a class, or no interaction
            index object.
        ValueError: If the order is out of range for the game.
    """
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
    return order
