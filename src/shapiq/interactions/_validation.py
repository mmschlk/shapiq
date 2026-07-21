from __future__ import annotations

import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapiq.interactions._types import Interaction, InteractionOrientation


def normalize_interaction(
    interaction: Sequence[int],
    *,
    orientation: InteractionOrientation,
    n_players: int,
) -> Interaction:
    """Validate and normalize a single interaction key."""
    players = tuple(_validate_player(player, n_players=n_players) for player in interaction)
    if len(set(players)) != len(players):
        msg = "interactions must not contain repeated players"
        raise ValueError(msg)
    if orientation == "undirected":
        return tuple(sorted(players))
    if orientation == "directed":
        return players
    msg = f"unsupported interaction orientation: {orientation!r}"
    raise ValueError(msg)


def validate_interaction_metadata(
    *,
    index_name: str,
    order: int,
    orientation: InteractionOrientation,
    n_players: int,
) -> None:
    """Validate shared interaction-index metadata."""
    if not isinstance(index_name, str):
        msg = "index_name must be a string"
        raise TypeError(msg)
    if not index_name:
        msg = "index_name must be a non-empty string"
        raise ValueError(msg)
    if isinstance(order, bool) or not isinstance(order, int):
        msg = f"order must be an integer, got {type(order).__name__}"
        raise TypeError(msg)
    if order < 0 or order > n_players:
        msg = "order must satisfy 0 <= order <= n_players"
        raise ValueError(msg)
    if orientation not in {"undirected", "directed"}:
        msg = f"unsupported interaction orientation: {orientation!r}"
        raise ValueError(msg)
    if orientation != "undirected":
        msg = f"{index_name} currently supports only undirected interactions"
        raise ValueError(msg)


def _validate_player(player: int, *, n_players: int) -> int:
    if isinstance(player, bool):
        msg = "player indices must be integers, not bools"
        raise TypeError(msg)
    try:
        position = operator.index(player)
    except TypeError:
        msg = f"player indices must be integers, got {type(player).__name__}"
        raise TypeError(msg) from None
    if position < 0 or position >= n_players:
        msg = f"player index {position} is out of range for {n_players} players"
        raise ValueError(msg)
    return position
