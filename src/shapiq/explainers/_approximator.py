from __future__ import annotations

import warnings
from abc import ABC
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
from shapiq.games import Game
from shapiq.sampling import ApproximationState, EmptyState, Sampler, SamplingState
from shapiq.sampling._state import coalition_keys

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.explanations import ExplanationArray
    from shapiq.interactions import InteractionIndex

_STALL_UNITS = 10


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

    state: ApproximationState
    sampler: Sampler
    deduplicate: bool
    bank: int
    spent: int
    units_done: int

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
        self.state = EmptyState()
        self.bank = 0
        self.spent = 0
        self.units_done = 0
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
        n_samples = self.state.n_samples
        return (
            f"{type(self).__name__}(interaction_index={self.interaction_index!r}, "
            f"order={self.order!r}, n_samples={n_samples!r}, bank={self.bank!r}, "
            f"spent={self.spent!r}, deduplicate={self.deduplicate!r})"
        )

    def sample(self, budget: int) -> Self:
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
        remaining = self.bank + budget
        fresh = not isinstance(self.state, SamplingState)
        seeds = self.n_seed_samples if fresh else 0
        if fresh and remaining < seeds:
            return self._evolve(bank=remaining)
        n_units = (remaining - seeds) // self.unit_rows
        if not fresh and n_units == 0:
            evidence = self._checkpoint(cast("SamplingState[Array]", self.state), remaining)
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
            evidence = cast("SamplingState[Array]", self.state).append(
                DenseCoalitionArray(masks),
                values,
            )
        bank = remaining - rows
        evidence = self._checkpoint(evidence, bank)
        return self._evolve(
            state=evidence,
            units_done=self.units_done + n_units,
            spent=self.spent + rows,
            bank=bank,
        )

    def _sample_deduplicated(self, budget: int) -> Self:
        """Spend budget on novel evaluations only, reusing stored values."""
        spent = self.spent
        units_done = self.units_done
        remaining = self.bank + budget
        if isinstance(self.state, SamplingState):
            evidence = cast("SamplingState[Array]", self.state)
        else:
            seeds = self.n_seed_samples
            if remaining < seeds:
                return self._evolve(bank=remaining)
            evidence, seed_spent = self._evaluate_seeds()
            spent += seed_spent
            remaining -= seed_spent
        known: dict[bytes, int] = {}
        packed = evidence.packed_keys()
        width = packed.shape[-1]
        blob = packed.tobytes()
        for index in range(packed.shape[0]):
            known.setdefault(blob[index * width : (index + 1) * width], index)
        unit_rows = self.unit_rows
        quiet_units = self._quiet_units
        exhaustive = 2**self.game.n_players
        exhausted = len(known) >= exhaustive
        while remaining > 0 and quiet_units < _STALL_UNITS and not exhausted:
            n_request = max(-(-remaining // unit_rows), 1)
            masks = self._unit_masks(n_request, first_unit=units_done)
            keys = coalition_keys(np.asarray(masks))
            base = evidence.n_samples
            novel_positions: list[int] = []
            state_duplicates: list[tuple[int, int]] = []
            batch_duplicates: list[tuple[int, int]] = []
            batch_ranks: dict[bytes, int] = {}
            charge = 0
            kept_units = 0
            for unit in range(n_request):
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
                if charge >= remaining or quiet_units >= _STALL_UNITS or exhausted:
                    break
            keep_rows = kept_units * unit_rows
            masks = masks[..., :keep_rows, :]
            values = self._stitch_values(
                masks,
                evidence,
                [p for p in novel_positions if p < keep_rows],
                [pair for pair in state_duplicates if pair[0] < keep_rows],
                [pair for pair in batch_duplicates if pair[0] < keep_rows],
            )
            evidence = evidence.append(DenseCoalitionArray(masks), values)
            for position in range(keep_rows):
                known.setdefault(keys[position], base + position)
            spent += charge
            remaining -= charge
            units_done += kept_units
        if remaining > 0 and (exhausted or quiet_units >= _STALL_UNITS):
            msg = (
                f"sampling stopped with {remaining} evaluations still banked: no novel "
                f"coalitions remain reachable (a game with {self.game.n_players} players "
                f"has at most 2**{self.game.n_players} distinct coalitions); evidence "
                "gathered so far remains valid"
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
            self.state._history_cuts  # noqa: SLF001 - finalizing cuts of a state this call created
            if isinstance(self.state, SamplingState)
            else ()
        )
        if evidence is self.state:
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
        start = self.units_done if first_unit is None else first_unit
        draws = self.sampler.draws(jnp.arange(start, start + n_units))
        rendered = self._render(draws)
        return _flatten_units(rendered)

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

    def _stitch_values(
        self,
        masks: Array,
        state: SamplingState[Array],
        novel_positions: list[int],
        state_duplicates: list[tuple[int, int]],
        batch_duplicates: list[tuple[int, int]],
    ) -> Array:
        """Evaluate novel coalitions and fill duplicates from stored values."""
        target_shape = self.game.target_shape
        value_shape = self.game.value_shape
        novel_values: Array | None = None
        if novel_positions:
            novel_index = jnp.asarray(novel_positions)
            novel_values = self._call_game(masks[..., novel_index, :])
        state_values = None
        if state_duplicates or not novel_positions:
            state_values = jnp.asarray(state.values)
        reference = novel_values if novel_values is not None else state_values
        if reference is None:  # unreachable: the seed block always precedes
            msg = "deduplicated sampling produced neither novel nor stored values"
            raise RuntimeError(msg)
        values = jnp.zeros(
            (*value_shape, *target_shape, int(masks.shape[-2])),
            dtype=reference.dtype,
        )
        if novel_values is not None:
            values = values.at[..., jnp.asarray(novel_positions)].set(novel_values)
        if state_duplicates and state_values is not None:
            positions = jnp.asarray([position for position, _ in state_duplicates])
            sources = jnp.asarray([source for _, source in state_duplicates])
            values = values.at[..., positions].set(state_values[..., sources])
        if batch_duplicates and novel_values is not None:
            positions = jnp.asarray([position for position, _ in batch_duplicates])
            ranks = jnp.asarray([rank for _, rank in batch_duplicates])
            values = values.at[..., positions].set(novel_values[..., ranks])
        return values

    def approximate(self, budget: int) -> ExplanationArray[Array]:
        """Sample a budget and return explanations."""
        return self.sample(budget).explain()

    def rollback(self, steps: int = 1) -> Self:
        """Return a previous approximator.

        Samplers are stateless, so rolling back only rewinds the evidence
        state and the derived counters; the checkpoint restores the banked
        remainder, so rolling back and resampling the same budgets replays
        the same evidence.
        """
        rolled = self.state.rollback(steps)
        return self._at_state(rolled)  # rollback(0) returns this approximator itself

    def history(
        self,
        *,
        reverse: bool = False,
        include_self: bool = True,
    ) -> list[Self]:
        """Return value-equivalent approximator history."""
        states = self.state.history(reverse=reverse, include_self=include_self)
        return [self._at_state(state) for state in states]

    def _at_state(self, state: ApproximationState) -> Self:
        """Return this approximator rewound to a historical state.

        Counters are recomputed from the evidence: units from the stored
        row count, spend from the distinct-coalition count under
        deduplication (duplicates were free) and the row count otherwise;
        the bank is forfeited.
        """
        if state is self.state:
            return self
        if not isinstance(state, SamplingState):
            return self._evolve(state=state, units_done=0, spent=0, bank=0, quiet_units=0)
        sampled = state.n_samples - self.n_seed_samples
        units_done = max(sampled // self.unit_rows, 0)
        spent = len(state.unique().counts) if self.deduplicate else state.n_samples
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
        clone.state = self.state if state is None else state
        clone.units_done = self.units_done if units_done is None else units_done
        clone.spent = self.spent if spent is None else spent
        clone.bank = self.bank if bank is None else bank
        clone._quiet_units = self._quiet_units if quiet_units is None else quiet_units  # noqa: SLF001
        return clone

    def _require_no_evidence_yet(self) -> NoReturn:
        """Raise the standard error for explaining without any evidence."""
        msg = (
            f"no samples yet: sample at least {self.min_budget} evaluations first; "
            "note that sample() returns a new approximator: "
            "`approximator = approximator.sample(budget)`"
        )
        raise InsufficientSamplesError(msg)


def _flatten_units(batch: Array) -> Array:
    """Merge a leading unit axis into the sample axis, preserving unit order."""
    stacked = jnp.moveaxis(batch, 0, -3)
    return stacked.reshape(*stacked.shape[:-3], -1, stacked.shape[-1])
