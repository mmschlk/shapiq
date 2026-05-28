from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapiq.interactions._types import Interaction, InteractionIndexName, InteractionOrientation


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
    interaction_index: InteractionIndexName,
    order: int,
    orientation: InteractionOrientation,
    n_players: int,
) -> None:
    """Validate shared interaction-index metadata."""
    if not isinstance(interaction_index, str):
        msg = "interaction_index must be a string"
        raise TypeError(msg)
    if interaction_index not in {"SV", "SII", "k-SII", "STII", "FSII"}:
        msg = f"unsupported interaction index: {interaction_index!r}"
        raise ValueError(msg)
    if isinstance(order, bool) or not isinstance(order, int):
        msg = "order must be an integer, not a bool"
        raise TypeError(msg)
    if order < 0 or order > n_players:
        msg = "order must satisfy 0 <= order <= n_players"
        raise ValueError(msg)
    if orientation not in {"undirected", "directed"}:
        msg = f"unsupported interaction orientation: {orientation!r}"
        raise ValueError(msg)
    if interaction_index == "SV" and order != 1:
        msg = "SV requires order == 1"
        raise ValueError(msg)
    if interaction_index in {"SV", "SII", "k-SII", "STII", "FSII"} and orientation != "undirected":
        msg = f"{interaction_index} currently supports only undirected interactions"
        raise ValueError(msg)


def _validate_player(player: int, *, n_players: int) -> int:
    if isinstance(player, bool) or not isinstance(player, int):
        msg = "player indices must be integers, not bools"
        raise TypeError(msg)
    if player < 0 or player >= n_players:
        msg = "player index out of range"
        raise ValueError(msg)
    return player
