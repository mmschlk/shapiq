from __future__ import annotations

from itertools import combinations, permutations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from shapiq.interactions._types import Interaction, InteractionOrientation


def iter_interactions(
    n_players: int,
    order: int,
    min_order: int = 0,
    orientation: InteractionOrientation = "undirected",
) -> Iterator[Interaction]:
    """Iterate interactions by increasing order."""
    if not 0 <= min_order <= order <= n_players:
        msg = "expected 0 <= min_order <= order <= n_players"
        raise ValueError(msg)
    if orientation not in {"undirected", "directed"}:
        msg = f"unsupported interaction orientation: {orientation!r}"
        raise ValueError(msg)
    players = range(n_players)
    for size in range(min_order, order + 1):
        if orientation == "undirected":
            yield from combinations(players, size)
        else:
            yield from permutations(players, size)
