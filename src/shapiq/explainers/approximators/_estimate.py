"""Coefficient assembly and evidence-derived counters for the engine."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from shapiq.sampling import Evidence, SampledEvidence

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


def trailing_quiet_units(evidence: Evidence, unit_rows: int, n_seed_samples: int) -> int:
    """Derive the stall counter: trailing whole units with no novel coalition.

    A row is novel exactly when it is its coalition's first occurrence in
    the stream (``row == key_index[key]``). Deriving the counter from the
    evidence — instead of carrying it — is what makes rollback and replay
    exact across a stall: the counter can never disagree with the stream.
    """
    if not isinstance(evidence, SampledEvidence):
        return 0
    index = evidence.key_index()
    keys = _row_keys(evidence)
    quiet = 0
    unit_end = evidence.n_samples
    while unit_end - unit_rows >= n_seed_samples:
        start = unit_end - unit_rows
        novel = any(index[keys[row]] == row for row in range(start, unit_end))
        if novel:
            break
        quiet += 1
        unit_end = start
    return quiet


def _row_keys(evidence: SampledEvidence) -> list[bytes]:
    packed = evidence.packed_keys()
    width = packed.shape[-1]
    blob = packed.tobytes()
    return [blob[start : start + width] for start in range(0, len(blob), width)]
