"""The sample loop as pure functions over ``(evidence, bank)``.

The engine is the process side of estimation: it spends a budget against a
frozen policy's configuration and returns grown evidence with the banked
remainder. Nothing here holds state — every counter the loop needs (units
drawn, the stall counter) is derived from the evidence at entry, which is
what makes any estimate an exact resume point: rollback and resample
replay bit-identically, stalls included.

Budgets are spent in whole units — the seed block once, then sampled
units — and the remainder is banked and spent first on the next call, so
budgets may be split freely across calls without changing the sampled
evidence. With deduplication enabled, only novel coalitions are charged
and the final unit may borrow from the bank (negative = borrowed, repaid
by the next call).

This module is the loop behind the policy's verbs and reads the policy's
private configuration by design (``SLF001`` is off for this file).
"""

from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np

from shapiq._shape import validate_int
from shapiq._valueaxes import to_leading
from shapiq.coalitions import DenseCoalitionArray
from shapiq.errors import SamplingStallWarning
from shapiq.explainers._deduplication import (
    STALL_UNITS,
    admit_units,
    stitch_values,
)
from shapiq.sampling import Evidence, SampledEvidence
from shapiq.sampling._evidence import coalition_keys

if TYPE_CHECKING:
    from jax import Array

    from shapiq.explainers._approximator import Approximator


def grow(
    policy: Approximator,
    evidence: Evidence,
    bank: int,
    budget: int,
) -> tuple[Evidence, int]:
    """Sample and evaluate additional coalitions for one policy.

    Args:
        policy: The frozen policy whose configuration drives the loop.
        evidence: The evidence gathered so far.
        bank: The banked budget remainder riding with the evidence.
        budget: Number of new coalition evaluations to spend.

    Returns:
        The grown evidence and the new banked remainder; the inputs are
        unchanged.

    Warns:
        SamplingStallWarning: If deduplication is enabled and the sampler
            stops producing novel coalitions.
    """
    validate_int("budget", budget)
    if budget == 0:
        return evidence, bank
    if policy.deduplicate:
        return _grow_deduplicated(policy, evidence, bank + budget)
    return _grow_plain(policy, evidence, bank + budget)


def banked_at(evidence: Evidence) -> int:
    """Return the banked remainder the evidence's last checkpoint recorded."""
    if not isinstance(evidence, SampledEvidence):
        return 0
    cuts = evidence._history_cuts
    return cuts[-1][1] if cuts else 0


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


def _grow_plain(
    policy: Approximator,
    evidence: Evidence,
    remaining: int,
) -> tuple[Evidence, int]:
    """Spend whole units without deduplication: every drawn row is charged."""
    fresh = not isinstance(evidence, SampledEvidence)
    seeds = policy.n_seed_samples if fresh else 0
    if fresh and remaining < seeds:
        return evidence, remaining
    n_units = (remaining - seeds) // policy.unit_rows
    if not fresh and n_units == 0:
        sampled = cast("SampledEvidence[Array]", evidence)
        return _with_checkpoint(sampled, remaining, base=evidence), remaining
    blocks: list[Array] = []
    if fresh:
        blocks.append(_seed_masks(policy))
    if n_units:
        blocks.append(_unit_masks(policy, n_units, first_unit=_units_stored(policy, evidence)))
    masks = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=-2)
    values = _call_game(policy, masks)
    if fresh:
        grown: SampledEvidence[Array] = SampledEvidence(
            coalitions=DenseCoalitionArray(masks),
            values=values,
            target_shape=policy.game.target_shape,
        )
    else:
        grown = cast("SampledEvidence[Array]", evidence).append(
            DenseCoalitionArray(masks),
            values,
        )
    bank = remaining - (seeds + n_units * policy.unit_rows)
    return _with_checkpoint(grown, bank, base=evidence), bank


def _grow_deduplicated(
    policy: Approximator,
    entry: Evidence,
    remaining: int,
) -> tuple[Evidence, int]:
    """Spend budget on novel evaluations only, reusing stored values.

    The admission policy lives in ``_deduplication``: whole units are
    admitted against the evidence's key index, only novel rows reach the
    game-call seam, and duplicate rows are stitched from values already
    computed. The stall counter is a pure function of the trailing unit
    sequence, recomputed at entry.
    """
    if isinstance(entry, SampledEvidence):
        evidence = cast("SampledEvidence[Array]", entry)
    else:
        seeds = policy.n_seed_samples
        if remaining < seeds:
            return entry, remaining
        seed_masks = _seed_masks(policy)
        evidence = SampledEvidence(
            coalitions=DenseCoalitionArray(seed_masks),
            values=_call_game(policy, seed_masks),
            target_shape=policy.game.target_shape,
        )
        remaining -= int(seed_masks.shape[-2])
    known = dict(evidence.key_index())
    unit_rows = policy.unit_rows
    units_done = _units_stored(policy, evidence)
    quiet_units = trailing_quiet_units(evidence, unit_rows, policy.n_seed_samples)
    leading_shape = (*policy.game.value_shape, *policy.game.target_shape)
    exhaustive = 2**policy.game.n_players
    exhausted = len(known) >= exhaustive
    while remaining > 0 and quiet_units < STALL_UNITS and not exhausted:
        n_request = max(-(-remaining // unit_rows), 1)
        masks = _unit_masks(policy, n_request, first_unit=units_done)
        keys = coalition_keys(np.asarray(masks))
        admission = admit_units(
            keys,
            known,
            unit_rows=unit_rows,
            remaining=remaining,
            quiet_units=quiet_units,
            exhaustive=exhaustive,
        )
        keep_rows = admission.kept_units * unit_rows
        masks = masks[..., :keep_rows, :]
        novel_values = (
            _call_game(policy, masks[..., jnp.asarray(admission.novel_positions), :])
            if admission.novel_positions
            else None
        )
        stored_values = (
            jnp.asarray(evidence.values)
            if admission.state_duplicates or novel_values is None
            else None
        )
        values = stitch_values(
            admission,
            novel_values,
            stored_values,
            leading_shape=leading_shape,
            n_rows=keep_rows,
        )
        base = evidence.n_samples
        evidence = evidence.append(DenseCoalitionArray(masks), values)
        for position in admission.novel_positions:
            known[keys[position]] = base + position
        quiet_units = admission.quiet_units
        exhausted = admission.exhausted
        remaining -= admission.charge
        units_done += admission.kept_units
    if remaining > 0 and (exhausted or quiet_units >= STALL_UNITS):
        if exhausted:
            msg = (
                f"sampling stopped with {remaining} evaluations still banked: every "
                f"distinct coalition of the {policy.game.n_players}-player game has "
                "been evaluated; evidence gathered so far remains valid"
            )
        else:
            msg = (
                f"sampling stopped with {remaining} evaluations still banked: the "
                f"sampler produced no novel coalition in {STALL_UNITS} consecutive "
                "units; evidence gathered so far remains valid"
            )
        warnings.warn(msg, SamplingStallWarning, stacklevel=4)
    return _with_checkpoint(evidence, remaining, base=entry), remaining


def _with_checkpoint(
    evidence: SampledEvidence[Array],
    bank: int,
    *,
    base: Evidence,
) -> SampledEvidence[Array]:
    """Record one history checkpoint for this sample call.

    A checkpoint is ``(n_samples, bank)``: per-iteration appends (and the
    seed block) collapse into one cut per call — ``base`` is the evidence
    the call entered with, whose cuts are kept — and the banked remainder
    rides along so rollback restores the exact resume point. Banking-only
    calls append a cut with the unchanged row count on a shallow copy, so
    every sample call is a resume point.
    """
    base_cuts = base._history_cuts if isinstance(base, SampledEvidence) else ()
    if evidence is base:
        evidence = copy(evidence)
    evidence._history_cuts = (*base_cuts, (evidence.n_samples, bank))
    return evidence


def _units_stored(policy: Approximator, evidence: Evidence) -> int:
    """Return how many whole units the evidence holds past the seed block.

    Whole-unit spending guarantees the stored rows are the seed block plus
    complete units, so the count doubles as the next draw index — units a
    stall rejected were never appended and are redrawn under the same
    index.
    """
    if not isinstance(evidence, SampledEvidence):
        return 0
    return max((evidence.n_samples - policy.n_seed_samples) // policy.unit_rows, 0)


def _seed_masks(policy: Approximator) -> Array:
    """Return the seed block: empty and grand coalition, then the prelude."""
    base = jnp.stack(
        [
            jnp.zeros(policy.game.n_players, dtype=bool),
            jnp.ones(policy.game.n_players, dtype=bool),
        ],
    )
    block = base if policy._prelude_masks is None else jnp.concatenate(
        [base, policy._prelude_masks],
        axis=-2,
    )
    return jnp.broadcast_to(
        block,
        (*policy.sampler.shared_target_shape, *block.shape),
    )


def _unit_masks(policy: Approximator, n_units: int, *, first_unit: int) -> Array:
    """Render whole sampled units into one coalition block."""
    draws = policy.sampler.draws(jnp.arange(first_unit, first_unit + n_units))
    rendered = policy._render(draws)
    return _flatten_units(rendered)


def _call_game(policy: Approximator, masks: Array) -> Array:
    """Evaluate the game and enter values in the canonical layout.

    This is the single seam where boundary values (broadcast targets, then
    samples, then value axes — the public game contract) become the
    canonical internal layout: value axes leading, sample axis last.
    Everything behind this seam computes on the canonical layout and never
    moves value axes again.
    """
    values = policy.game(DenseCoalitionArray(masks))
    return to_leading(jnp.asarray(values), len(policy.game.value_shape))


def _flatten_units(batch: Array) -> Array:
    """Merge a leading unit axis into the sample axis, preserving unit order."""
    stacked = jnp.moveaxis(batch, 0, -3)
    return stacked.reshape(*stacked.shape[:-3], -1, stacked.shape[-1])
