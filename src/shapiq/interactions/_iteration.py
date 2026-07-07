from __future__ import annotations

from functools import cache
from itertools import combinations, permutations
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jax import Array

    from shapiq.interactions._types import Interaction, InteractionOrientation


@cache
def interaction_masks(n_players: int, size: int) -> Array:
    """Return dense member masks of all size-``size`` interactions, lexicographic."""
    return jnp.asarray(
        [
            [player in members for player in range(n_players)]
            for members in combinations(range(n_players), size)
        ],
    )


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
