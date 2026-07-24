"""Assembly of per-order coefficient blocks into a surrogate's sparse terms."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array


def leading_blocks_to_terms(
    attributions: Mapping[int, Array],
    n_players: int,
    empty: Array | None,
) -> tuple[tuple[frozenset[int], ...], np.ndarray]:
    """Assemble per-order coefficient blocks into aligned terms and values.

    Args:
        attributions: Per-interaction-size blocks in the canonical leading
            layout ``(*value_shape, *target_shape, n_interactions_of_size)``,
            interactions in ``combinations`` order.
        n_players: Number of players the interactions range over.
        empty: The empty interaction's coefficient block
            ``(*value_shape, *target_shape)`` or ``None`` when the family
            declares no order-0 slot (and no size-0 block is given).

    Returns:
        Terms and the concatenated host float64 coefficient array.
    """
    terms: list[frozenset[int]] = []
    blocks: list[np.ndarray] = []
    if empty is not None and 0 not in attributions:
        terms.append(frozenset())
        blocks.append(np.asarray(empty, dtype=np.float64)[..., None])
    for size in sorted(attributions):
        block = np.asarray(attributions[size], dtype=np.float64)
        if size == 0:
            terms.append(frozenset())
            blocks.append(block.reshape(*block.shape[:-1], 1) if block.shape[-1:] == (1,) else block[..., None])
            continue
        terms.extend(frozenset(members) for members in combinations(range(n_players), size))
        blocks.append(block)
    if not blocks:
        return (), np.zeros(0)
    return tuple(terms), np.concatenate(blocks, axis=-1)
