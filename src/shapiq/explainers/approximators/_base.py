from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import copy
from typing import TYPE_CHECKING, NoReturn, Self, cast

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq._shape import ensure_bool, logical_size, validate_int
from shapiq.coalitions import DenseCoalitionArray
from shapiq.errors import InsufficientSamplesError, SamplingStallWarning
from shapiq.explainers._base import Explainer
from shapiq.explainers._valueaxes import to_leading
from shapiq.explainers.approximators._deduplication import (
    STALL_UNITS,
    admit_units,
    stitch_values,
)
from shapiq.explainers.approximators._estimate import (
    Estimate,
    leading_blocks_to_terms,
    trailing_quiet_units,
)
from shapiq.games import Game
from shapiq.sampling import ApproximationState, EmptyState, Sampler, SamplingState
from shapiq.sampling._state import coalition_keys

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.interactions import InteractionIndex


class Approximator(Explainer[Array, Game[Array]], ABC):
    """Base abstraction for sampling-based explainers: one loop, one seam.

    An approximator has a sampler and houses the approximator logic. The
    sampler is a stateless draw value; the approximator owns everything
    downstream in its single sample loop: it evaluates the deterministic
    seed block once (the empty and grand coalition plus any family
    prelude), spends budgets in whole units, banks the remainder as one
    integer, renders draws through the family's expansion, and evaluates
    through the game-call seam. Budgets are denominated in game
    evaluations; whole-unit spending with a banked remainder keeps sampled
    streams invariant under budget splits, and evaluations that could not
    inform an estimate are never made.

    With deduplication enabled, every distinct coalition is evaluated on
    the game at most once: repeated coalitions reuse stored values, only
    novel evaluations are charged, and the final unit may overshoot into
    the bank (negative = borrowed, repaid by the next call). If the
    sampler stops producing novel coalitions, the remaining budget stays
    banked and a ``SamplingStallWarning`` is issued.
    """

    sampler: Sampler
    deduplicate: bool

    def __init__(
        self,
        game: Game[Array],
        sampler: Sampler,
        index: InteractionIndex,
        *,
        render: Callable[[Array], Array],
        unit_length: int,
        prelude_masks: Array | None = None,
        deduplicate: bool = False,
    ) -> None:
        """Initialize an approximator without evaluating the game.

        Args:
            game: Game to explain.
            sampler: Stateless draw source matching the game's players and
                target shape.
            index: The interaction index the subclass estimates.
            render: The family's expansion from draws to per-unit coalition
                masks (identity rows for coalition samplers).
            unit_length: Coalition rows one draw materializes into.
            prelude_masks: Deterministic seed masks following the empty and
                grand coalition, or ``None``.
            deduplicate: Whether to evaluate each distinct coalition at most
                once.
        """
        super().__init__(game, index)
        if sampler.n_players != game.n_players:
            msg = "sampler and game use different numbers of players"
            raise ValueError(msg)
        if sampler.target_shape != game.target_shape:
            msg = "sampler and game use different target shapes"
            raise ValueError(msg)
        validate_int("unit_length", unit_length, minimum=1)
        self.sampler = sampler
        self._state: ApproximationState = EmptyState()
        self._bank = 0
        self._spent = 0
        self._units_done = 0
        self._render = render
        self._unit_length = unit_length
        self._prelude_masks = prelude_masks
        self._quiet_units = 0
        self.deduplicate = ensure_bool("deduplicate", deduplicate)
        if self.deduplicate and logical_size(sampler.shared_target_shape) != 1:
            msg = (
                "deduplicate=True requires the same coalitions to be sampled for every "
                "explanation target; pass share_samples=True (or share the selected axes)"
            )
            raise ValueError(msg)

    @property
    def n_seed_samples(self) -> int:
        """Return the seed block length: empty and grand coalition, then prelude."""
        if self._prelude_masks is None:
            return 2
        return 2 + int(self._prelude_masks.shape[-2])

    @property
    def unit_rows(self) -> int:
        """Return the coalition rows one sampled unit contributes."""
        return self._unit_length * self.sampler.draws_per_unit

    @property
    def min_budget(self) -> int:
        """Return the floor below which ``explain()`` cannot succeed.

        The first explanation needs the seed block plus one completed
        sampled unit. It is a floor, not a guarantee: whether the drawn
        coalitions carry enough evidence is method-specific (interaction
        coverage for permutation walks, identification for the
        regressions), and ``explain()`` raises ``InsufficientSamplesError``
        stating the shortfall while they do not.
        """
        return self.n_seed_samples + self.unit_rows

    def __repr__(self) -> str:
        """Return a concise representation for the rebind workflow."""
        n_samples = self._state.n_samples
        return (
            f"{type(self).__name__}(interaction_index={self.interaction_index!r}, "
            f"order={self.order!r}, n_samples={n_samples!r}, bank={self._bank!r}, "
            f"spent={self._spent!r}, deduplicate={self.deduplicate!r})"
        )

    def _grow(self, budget: int) -> Self:
        """Sample and evaluate additional coalitions, deduplicating if enabled.

        Budgets are spent in whole units — the seed block once, then
        sampled units — and the remainder is banked and spent first on the
        next call, so budgets may be split freely across calls without
        changing the sampled evidence. With deduplication enabled, only
        novel coalitions are charged and the final unit may borrow from the
        bank.

        Args:
            budget: Number of new coalition evaluations to spend.

        Returns:
            A new approximator whose state includes the sampled evidence;
            this approximator is unchanged.

        Warns:
            SamplingStallWarning: If deduplication is enabled and the
                sampler stops producing novel coalitions.
        """
        validate_int("budget", budget)
        if budget == 0:
            return self
        if self.deduplicate:
            return self._sample_deduplicated(budget)
        remaining = self._bank + budget
        fresh = not isinstance(self._state, SamplingState)
        seeds = self.n_seed_samples if fresh else 0
        if fresh and remaining < seeds:
            return self._evolve(bank=remaining)
        n_units = (remaining - seeds) // self.unit_rows
        if not fresh and n_units == 0:
            evidence = self._checkpoint(cast("SamplingState[Array]", self._state), remaining)
            return self._evolve(state=evidence, bank=remaining)
        blocks: list[Array] = []
        if fresh:
            blocks.append(self._seed_masks())
        if n_units:
            blocks.append(self._unit_masks(n_units))
        masks = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=-2)
        values = self._call_game(masks)
        rows = seeds + n_units * self.unit_rows
        if fresh:
            evidence: SamplingState[Array] = SamplingState(
                coalitions=DenseCoalitionArray(masks),
                values=values,
                target_shape=self.game.target_shape,
            )
        else:
            evidence = cast("SamplingState[Array]", self._state).append(
                DenseCoalitionArray(masks),
                values,
            )
        bank = remaining - rows
        evidence = self._checkpoint(evidence, bank)
        return self._evolve(
            state=evidence,
            units_done=self._units_done + n_units,
            spent=self._spent + rows,
            bank=bank,
        )

    def _sample_deduplicated(self, budget: int) -> Self:
        """Spend budget on novel evaluations only, reusing stored values.

        The admission policy lives in ``_deduplication``: whole units are
        admitted against the state's key index, only novel rows reach the
        game-call seam, and duplicate rows are stitched from values
        already computed.
        """
        spent = self._spent
        units_done = self._units_done
        remaining = self._bank + budget
        if isinstance(self._state, SamplingState):
            evidence = cast("SamplingState[Array]", self._state)
        else:
            seeds = self.n_seed_samples
            if remaining < seeds:
                return self._evolve(bank=remaining)
            evidence, seed_spent = self._evaluate_seeds()
            spent += seed_spent
            remaining -= seed_spent
        known = dict(evidence.key_index())
        unit_rows = self.unit_rows
        quiet_units = self._quiet_units
        leading_shape = (*self.game.value_shape, *self.game.target_shape)
        exhaustive = 2**self.game.n_players
        exhausted = len(known) >= exhaustive
        while remaining > 0 and quiet_units < STALL_UNITS and not exhausted:
            n_request = max(-(-remaining // unit_rows), 1)
            masks = self._unit_masks(n_request, first_unit=units_done)
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
                self._call_game(masks[..., jnp.asarray(admission.novel_positions), :])
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
            spent += admission.charge
            remaining -= admission.charge
            units_done += admission.kept_units
        if remaining > 0 and (exhausted or quiet_units >= STALL_UNITS):
            if exhausted:
                msg = (
                    f"sampling stopped with {remaining} evaluations still banked: every "
                    f"distinct coalition of the {self.game.n_players}-player game has "
                    "been evaluated; evidence gathered so far remains valid"
                )
            else:
                msg = (
                    f"sampling stopped with {remaining} evaluations still banked: the "
                    f"sampler produced no novel coalition in {STALL_UNITS} consecutive "
                    "units; evidence gathered so far remains valid"
                )
            warnings.warn(msg, SamplingStallWarning, stacklevel=3)
        evidence = self._checkpoint(evidence, remaining)
        return self._evolve(
            state=evidence,
            units_done=units_done,
            spent=spent,
            bank=remaining,
            quiet_units=quiet_units,
        )

    def _checkpoint(self, evidence: SamplingState[Array], bank: int) -> SamplingState[Array]:
        """Record one history checkpoint for this sample call.

        A checkpoint is ``(n_samples, bank)``: per-iteration appends (and
        the seed block) collapse into one cut per call, and the banked
        remainder rides along so rollback restores the exact resume point.
        Banking-only calls append a cut with the unchanged row count on a
        shallow state copy, so every sample call is a resume point.
        """
        base_cuts = (
            self._state._history_cuts  # noqa: SLF001 - finalizing cuts of a state this call created
            if isinstance(self._state, SamplingState)
            else ()
        )
        if evidence is self._state:
            evidence = copy(evidence)
        evidence._history_cuts = (*base_cuts, (evidence.n_samples, bank))  # noqa: SLF001
        return evidence

    def _evaluate_seeds(self) -> tuple[SamplingState[Array], int]:
        """Evaluate the seed block once and open the evidence state."""
        seed_masks = self._seed_masks()
        values = self._call_game(seed_masks)
        state: SamplingState[Array] = SamplingState(
            coalitions=DenseCoalitionArray(seed_masks),
            values=values,
            target_shape=self.game.target_shape,
        )
        return state, int(seed_masks.shape[-2])

    def _seed_masks(self) -> Array:
        """Return the seed block: empty and grand coalition, then the prelude."""
        base = jnp.stack(
            [
                jnp.zeros(self.game.n_players, dtype=bool),
                jnp.ones(self.game.n_players, dtype=bool),
            ],
        )
        block = base if self._prelude_masks is None else jnp.concatenate(
            [base, self._prelude_masks],
            axis=-2,
        )
        return jnp.broadcast_to(
            block,
            (*self.sampler.shared_target_shape, *block.shape),
        )

    def _unit_masks(self, n_units: int, first_unit: int | None = None) -> Array:
        """Render whole sampled units into one coalition block."""
        start = self._units_done if first_unit is None else first_unit
        draws = self.sampler.draws(jnp.arange(start, start + n_units))
        rendered = self._render(draws)
        return _flatten_units(rendered)

    @abstractmethod
    def _estimate_parts(self) -> tuple[dict[int, Array], Array | None]:
        """Build per-order coefficient blocks and the empty slot's value."""
        ...

    def _call_game(self, masks: Array) -> Array:
        """Evaluate the game and enter values in the canonical layout.

        This is the single seam where boundary values (broadcast targets,
        then samples, then value axes — the public game contract) become
        the canonical internal layout: value axes leading, sample axis
        last. Everything behind this seam computes on the canonical layout
        and never moves value axes again.
        """
        values = self.game(DenseCoalitionArray(masks))
        return to_leading(jnp.asarray(values), len(self.game.value_shape))

    def estimate(self, budget: int) -> Estimate:
        """Estimate the game from scratch: spend a budget, return the carry.

        The returned :class:`Estimate` is inert — a game-with-provenance;
        continue it with ``refine`` on this (frozen) policy.
        """
        value_shape = tuple(self.game.value_shape)
        target_shape = tuple(self.game.target_shape)
        fresh = Estimate(
            terms=(),
            values=np.zeros((*value_shape, *target_shape, 0)),
            n_players=self.game.n_players,
            evidence=EmptyState(),
            bank=0,
            index=self.index,
            deduplicated=self.deduplicate,
            target_shape=target_shape,
            value_shape=value_shape,
            unready_reason="no evidence yet: refine this estimate with a budget first",
        )
        return self.refine(fresh, budget)

    @property
    def _fingerprint(self) -> tuple[object, ...]:
        """Return the policy's structural identity for carry checks."""
        return (type(self).__name__, self.unit_rows, self.n_seed_samples, self.deduplicate)

    def _require_own_carry(self, carry: Estimate) -> None:
        if carry.fingerprint is not None and carry.fingerprint != self._fingerprint:
            msg = (
                f"this estimate was produced by {carry.fingerprint!r} and cannot be "
                f"continued by {self._fingerprint!r}: evidence rows would be "
                "reinterpreted under the wrong unit convention; build the intended "
                "policy, or derive from raw evidence with at_evidence()"
            )
            raise ValueError(msg)

    def refine(self, carry: Estimate, budget: int) -> Estimate:
        """Spend more budget on an estimate and return the grown carry.

        Counters are derived from the carried evidence (the stall counter
        included — it is a pure function of the unit sequence), so any
        estimate is an exact resume point: rollback and resample replay
        bit-identically, stalls included. Refining another policy's carry
        raises: the fingerprint says whose it is.
        """
        self._require_own_carry(carry)
        worker = self._at_state(carry.evidence)._evolve(  # noqa: SLF001 - transitional shim onto the loop
            bank=carry.bank,
            quiet_units=(
                trailing_quiet_units(carry.evidence, self.unit_rows, self.n_seed_samples)
                if self.deduplicate
                else 0
            ),
        )
        return self._as_estimate(worker._grow(budget))  # noqa: SLF001 - the loop behind the verb

    def at_evidence(self, evidence: ApproximationState, bank: int | None = None) -> Estimate:
        """Return the estimate a policy derives from given evidence.

        ``bank`` defaults to the banked remainder the evidence's last
        checkpoint recorded; rolling back an estimate is
        ``policy.at_evidence(carry.evidence.rollback(steps))``.
        """
        worker = self._at_state(evidence)
        if bank is not None:
            worker = worker._evolve(bank=bank)  # noqa: SLF001 - transitional shim
        return self._as_estimate(worker)

    def _as_estimate(self, worker: Approximator) -> Estimate:
        value_shape = tuple(self.game.value_shape)
        target_shape = tuple(self.game.target_shape)
        reason: str | None = None
        try:
            attributions, empty = worker._estimate_parts()
            terms, coefficients = leading_blocks_to_terms(
                attributions,
                self.game.n_players,
                empty,
            )
        except InsufficientSamplesError as error:
            reason = str(error)
            terms = ()
            coefficients = np.zeros((*value_shape, *target_shape, 0))
        return Estimate(
            terms=terms,
            values=coefficients,
            n_players=self.game.n_players,
            evidence=worker._state,
            bank=worker._bank,
            index=self.index,
            deduplicated=self.deduplicate,
            target_shape=target_shape,
            value_shape=value_shape,
            unready_reason=reason,
            fingerprint=self._fingerprint,
        )

    def _at_state(self, state: ApproximationState) -> Self:
        """Return this approximator rewound to a historical state.

        Counters are recomputed from the evidence: units from the stored
        row count, spend from the distinct-coalition count under
        deduplication (duplicates were free) and the row count otherwise;
        the bank is forfeited.
        """
        if state is self._state:
            return self
        if not isinstance(state, SamplingState):
            return self._evolve(state=state, units_done=0, spent=0, bank=0, quiet_units=0)
        sampled = state.n_samples - self.n_seed_samples
        units_done = max(sampled // self.unit_rows, 0)
        spent = len(state.key_index()) if self.deduplicate else state.n_samples
        bank = state._history_cuts[-1][1]  # noqa: SLF001 - the checkpoint carries the bank
        return self._evolve(state=state, units_done=units_done, spent=spent, bank=bank, quiet_units=0)

    def _evolve(
        self,
        *,
        state: ApproximationState | None = None,
        units_done: int | None = None,
        spent: int | None = None,
        bank: int | None = None,
        quiet_units: int | None = None,
    ) -> Self:
        clone = copy(self)
        # process state is private and evolves only on transient loop clones
        clone._state = self._state if state is None else state  # noqa: SLF001
        clone._units_done = self._units_done if units_done is None else units_done  # noqa: SLF001
        clone._spent = self._spent if spent is None else spent  # noqa: SLF001
        clone._bank = self._bank if bank is None else bank  # noqa: SLF001
        clone._quiet_units = self._quiet_units if quiet_units is None else quiet_units  # noqa: SLF001
        return clone

    def _require_no_evidence_yet(self) -> NoReturn:
        """Raise the standard error for explaining without any evidence."""
        msg = (
            f"no evidence yet: estimate with at least {self.min_budget} evaluations first; "
            "note that policies are frozen and return the estimate: "
            "`estimate = policy.estimate(budget)`"
        )
        raise InsufficientSamplesError(msg)


def _flatten_units(batch: Array) -> Array:
    """Merge a leading unit axis into the sample axis, preserving unit order."""
    stacked = jnp.moveaxis(batch, 0, -3)
    return stacked.reshape(*stacked.shape[:-3], -1, stacked.shape[-1])
