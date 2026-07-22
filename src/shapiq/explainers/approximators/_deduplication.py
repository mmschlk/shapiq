"""The deduplicating admission policy for the sample loop.

Deduplication admits rendered units against the evidence already stored:
each coalition row is classified as novel, a duplicate of a stored row,
or a duplicate of an earlier row in the same batch. Only whole units are
admitted and only novel rows are charged; every stopping rule is a pure
function of the unit sequence and cumulative counters — never of the
batch size — so stored streams stay invariant under budget splits. The
game call itself stays at the approximator's ``_call_game`` seam: the
policy decides which rows reach it (``admit_units``) and assembles the
duplicate rows from values already computed (``stitch_values``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array

STALL_UNITS = 10
"""Consecutive units without a novel coalition before sampling stalls."""


class UnitAdmission(NamedTuple):
    """One batch's admission verdict against the stored evidence.

    Row positions count within the admitted rows; stream positions count
    within the stored evidence; novel ranks count within the batch's novel
    rows in admission order.
    """

    kept_units: int
    """Whole units admitted, counted from the front of the batch."""

    charge: int
    """Novel rows among the admitted units — the budget actually spent."""

    quiet_units: int
    """Updated run of consecutive units without a novel coalition."""

    exhausted: bool
    """Whether every distinct coalition of the game is now known."""

    novel_positions: list[int]
    """Row positions that must reach the game."""

    state_duplicates: list[tuple[int, int]]
    """``(row position, stream position)`` pairs filled from stored values."""

    batch_duplicates: list[tuple[int, int]]
    """``(row position, novel rank)`` pairs filled from this batch's values."""


def admit_units(
    keys: list[bytes],
    known: Mapping[bytes, int],
    *,
    unit_rows: int,
    remaining: int,
    quiet_units: int,
    exhaustive: int,
) -> UnitAdmission:
    """Scan a rendered batch unit by unit and admit whole units.

    Args:
        keys: Coalition identity per batch row, unit-major.
        known: First stream position per distinct stored coalition.
        unit_rows: Coalition rows one sampled unit contributes.
        remaining: Budget available, in game evaluations.
        quiet_units: Run of consecutive novel-free units carried in.
        exhaustive: Total number of distinct coalitions the game has.

    Returns:
        The admission verdict; the batch's tail beyond ``kept_units`` was
        never inspected and must be discarded by the caller.
    """
    novel_positions: list[int] = []
    state_duplicates: list[tuple[int, int]] = []
    batch_duplicates: list[tuple[int, int]] = []
    batch_ranks: dict[bytes, int] = {}
    charge = 0
    kept_units = 0
    exhausted = False
    for unit in range(len(keys) // unit_rows):
        unit_novel = 0
        for row in range(unit_rows):
            position = unit * unit_rows + row
            key = keys[position]
            if key in batch_ranks:
                batch_duplicates.append((position, batch_ranks[key]))
            elif key in known:
                state_duplicates.append((position, known[key]))
            else:
                batch_ranks[key] = len(novel_positions)
                novel_positions.append(position)
                unit_novel += 1
        charge += unit_novel
        quiet_units = 0 if unit_novel else quiet_units + 1
        kept_units = unit + 1
        # every stopping rule is a pure function of the unit sequence
        # and cumulative counters, never of the batch size, so stored
        # streams stay invariant under budget splits even here
        if len(known) + len(batch_ranks) >= exhaustive:
            exhausted = True
        if charge >= remaining or quiet_units >= STALL_UNITS or exhausted:
            break
    return UnitAdmission(
        kept_units=kept_units,
        charge=charge,
        quiet_units=quiet_units,
        exhausted=exhausted,
        novel_positions=novel_positions,
        state_duplicates=state_duplicates,
        batch_duplicates=batch_duplicates,
    )


def stitch_values(
    admission: UnitAdmission,
    novel_values: Array | None,
    stored_values: Array | None,
    *,
    leading_shape: tuple[int, ...],
    n_rows: int,
) -> Array:
    """Assemble the admitted rows' values from novel and stored evaluations.

    Args:
        admission: The verdict whose positions align with the values.
        novel_values: Evaluated novel rows in admission order, or ``None``
            when the batch held none.
        stored_values: The stored evidence values, or ``None`` when no
            admitted row duplicates them.
        leading_shape: The canonical layout's leading axes (value axes,
            then target axes).
        n_rows: Admitted coalition rows to assemble.

    Returns:
        Values for every admitted row in the canonical layout.
    """
    reference = novel_values if novel_values is not None else stored_values
    if reference is None:  # unreachable: the seed block always precedes
        msg = "deduplicated sampling produced neither novel nor stored values"
        raise RuntimeError(msg)
    values = jnp.zeros((*leading_shape, n_rows), dtype=reference.dtype)
    if novel_values is not None:
        values = values.at[..., jnp.asarray(admission.novel_positions)].set(novel_values)
    if admission.state_duplicates and stored_values is not None:
        positions = jnp.asarray([position for position, _ in admission.state_duplicates])
        sources = jnp.asarray([source for _, source in admission.state_duplicates])
        values = values.at[..., positions].set(stored_values[..., sources])
    if admission.batch_duplicates and novel_values is not None:
        positions = jnp.asarray([position for position, _ in admission.batch_duplicates])
        ranks = jnp.asarray([rank for _, rank in admission.batch_duplicates])
        values = values.at[..., positions].set(novel_values[..., ranks])
    return values
