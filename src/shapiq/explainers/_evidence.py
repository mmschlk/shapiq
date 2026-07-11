from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NoReturn, cast

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq._shape import ensure_bool, logical_size, validate_int
from shapiq.coalitions import DenseCoalitionArray
from shapiq.errors import InsufficientSamplesError, SamplingStallWarning
from shapiq.explainers._approximator import Approximator
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.games import Game
from shapiq.sampling import ApproximationState, Sampler, SamplingState

if TYPE_CHECKING:
    from shapiq.coalitions import CoalitionArray


class EvidenceApproximator(
    Approximator[Array, Game[Array], ApproximationState, Sampler],
):
    """Shared machinery for approximators storing raw sampled evidence.

    Approximators start from an empty state; the sampler emits its
    deterministic seed block (empty and grand coalition, plus index-specific
    evaluations) at the start of the first sampled budget, so construction
    never evaluates the game. Pending samples of an unfinished unit stay in
    the state but are masked by ``explain()``.

    With deduplication enabled, every distinct coalition is evaluated on the
    game at most once: repeated coalitions reuse stored values, only novel
    evaluations consume budget, and the extra units are free. The estimate is
    identical to the non-deduplicated one over the same units. If the sampler
    stops producing novel coalitions, the remaining budget stays unspent and
    a ``SamplingStallWarning`` is issued.
    """

    deduplicate: bool
    _dedup_keys: tuple[int, int, dict[bytes, int]] | None

    @property
    def min_budget(self) -> int:
        """Return the floor below which ``explain()`` cannot succeed.

        The first explanation needs the seed block plus one completed sampled
        unit; this property saves users the arithmetic over the sampler's
        ``n_seed_samples`` and ``sampling_quantum``. It is a floor, not a
        guarantee: whether the drawn coalitions carry enough evidence is
        method-specific (interaction coverage for permutation walks,
        identification for the regressions), and ``explain()`` raises
        ``InsufficientSamplesError`` stating the shortfall while they do not.
        """
        return self.sampler.n_seed_samples + self.sampler.sampling_quantum

    def __repr__(self) -> str:
        """Return a concise representation for the rebind workflow."""
        n_samples = getattr(self.state, "n_samples", 0)
        return (
            f"{type(self).__name__}(interaction_index={self.interaction_index!r}, "
            f"order={self.order!r}, n_samples={n_samples!r}, "
            f"n_pending_samples={self.sampler.n_pending_samples!r}, "
            f"deduplicate={self.deduplicate!r})"
        )

    def _init_deduplication(self, *, deduplicate: bool) -> None:
        """Validate and store the deduplication policy."""
        self.deduplicate = ensure_bool("deduplicate", deduplicate)
        self._dedup_keys = None
        if self.deduplicate and logical_size(self.sampler.shared_target_shape) != 1:
            msg = (
                "deduplicate=True requires the same coalitions to be sampled for every "
                "explanation target; pass share_samples=True (or share the selected axes)"
            )
            raise ValueError(msg)

    def sample(
        self,
        budget: int,
    ) -> Approximator[Array, Game[Array], ApproximationState, Sampler]:
        """Sample and evaluate additional coalitions, deduplicating if enabled.

        The budget is spent exactly, starting with the one-time seed block on
        the first call. Units cut short by the budget stay pending and resume
        on the next call, so budgets may be split freely across calls without
        changing the sampled evidence.

        Args:
            budget: Number of new coalition evaluations to spend. With
                deduplication enabled, only novel coalitions count and
                repeated coalitions reuse stored values free of charge.

        Returns:
            A new approximator whose state includes the sampled evidence;
            this approximator is unchanged.

        Warns:
            SamplingStallWarning: If deduplication is enabled and the sampler
                stops producing novel coalitions before the budget is spent.
        """
        if not self.deduplicate:
            return super().sample(budget)
        validate_int("budget", budget)
        if budget == 0:
            return self
        return self._sample_deduplicated(budget)

    def _sample_deduplicated(
        self,
        budget: int,
    ) -> Approximator[Array, Game[Array], ApproximationState, Sampler]:
        """Spend budget on novel evaluations only, reusing stored values."""
        known = self._known_coalitions()
        sampler = self.sampler
        stall_limit = 10 * sampler.sampling_quantum
        chunks: list[Array] = []
        novel_positions: list[int] = []
        state_duplicates: list[tuple[int, int]] = []
        batch_duplicates: list[tuple[int, int]] = []
        batch_ranks: dict[bytes, int] = {}
        position = 0
        raw_since_novel = 0
        remaining = budget
        while remaining > 0:
            coalitions, sampler = sampler.sample(self.state, remaining)
            dense = jnp.asarray(coalitions.to_dense())
            chunks.append(dense)
            found = _classify_chunk(
                _packed_rows(dense),
                position,
                known=known,
                batch_ranks=batch_ranks,
                novel_positions=novel_positions,
                state_duplicates=state_duplicates,
                batch_duplicates=batch_duplicates,
            )
            position += int(dense.shape[-2])
            remaining -= found
            raw_since_novel = 0 if found else raw_since_novel + int(dense.shape[-2])
            if remaining > 0 and raw_since_novel >= stall_limit:
                msg = (
                    f"sampling stopped after {budget - remaining} of {budget} requested "
                    "evaluations: no novel coalitions were found (a game with "
                    f"{self.game.n_players} players has at most 2**{self.game.n_players} "
                    "distinct coalitions); evidence gathered so far remains valid"
                )
                warnings.warn(msg, SamplingStallWarning, stacklevel=3)
                break
        masks = jnp.concatenate(chunks, axis=-2)
        values = self._stitch_values(masks, novel_positions, state_duplicates, batch_duplicates)
        next_state = self._append_state(DenseCoalitionArray(masks), values)
        next_history = self._next_sampler_history(sampler)
        base = self.state.n_samples if isinstance(self.state, SamplingState) else 0
        for key, rank in batch_ranks.items():
            known[key] = base + novel_positions[rank]
        evolved = cast(
            "EvidenceApproximator",
            self._replace(state=next_state, sampler=sampler, sampler_history=next_history),
        )
        evolved._dedup_keys = (next_state.n_samples, len(known), known)  # noqa: SLF001 - evolving a copy of self
        return evolved

    def _known_coalitions(self) -> dict[bytes, int]:
        """Map every stored coalition to its first sample index.

        The map is carried forward across sample calls: the approximator at
        the tip of a sampling chain extends it in place, a branch (sampling
        twice from the same approximator) detects the foreign extension via
        the recorded length and copies its own entries, and any other state
        change (rollback, history) misses the sample-count token and rebuilds
        from the stored coalitions.
        """
        if not isinstance(self.state, SamplingState):
            return {}
        n_samples = self.state.n_samples
        if self._dedup_keys is not None:
            cached_samples, cached_length, cached = self._dedup_keys
            if cached_samples == n_samples:
                if cached_length == len(cached):
                    return cached
                return {key: index for key, index in cached.items() if index < n_samples}
        packed = _packed_rows(jnp.asarray(self.state.coalitions.to_dense()))
        unique_rows, first_indices = np.unique(packed, axis=0, return_index=True)
        return {
            row.tobytes(): int(index)
            for row, index in zip(unique_rows, first_indices, strict=True)
        }

    def _stitch_values(
        self,
        masks: Array,
        novel_positions: list[int],
        state_duplicates: list[tuple[int, int]],
        batch_duplicates: list[tuple[int, int]],
    ) -> Array:
        """Evaluate novel coalitions and fill duplicates from stored values."""
        target_shape = self.game.target_shape
        value_shape = self.game.value_shape
        n_value_axes = len(value_shape)
        novel_values: Array | None = None
        if novel_positions:
            novel_index = jnp.asarray(novel_positions)
            novel_values = to_leading(
                jnp.asarray(self.game(DenseCoalitionArray(masks[..., novel_index, :]))),
                n_value_axes,
            )
        state_values = None
        if isinstance(self.state, SamplingState):
            state_values = to_leading(
                jnp.asarray(cast("SamplingState[Array]", self.state).values),
                n_value_axes,
            )
        reference = novel_values if novel_values is not None else state_values
        if reference is None:  # unreachable: a first call always yields novel seeds
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
        return to_trailing(values, n_value_axes)

    def _append_state(self, coalitions: CoalitionArray, values: Array) -> SamplingState[Array]:
        """Append sampled coalitions, creating the evidence state on first use.

        Value shapes are validated at the game boundary; states store values
        with the sample axis at the target-shape position and any value axes
        trailing.
        """
        checked_values = jnp.asarray(values)
        if isinstance(self.state, SamplingState):
            return cast("SamplingState[Array]", self.state).append(coalitions, checked_values)
        return SamplingState(
            coalitions=coalitions,
            values=checked_values,
            target_shape=self.game.target_shape,
            track_history=self.state.track_history,
        )

    def _next_sampler_history(
        self,
        next_sampler: Sampler,
    ) -> tuple[Sampler, ...] | None:
        """Start sampler history at the first evidence state."""
        if self.state.track_history and not isinstance(self.state, SamplingState):
            return (next_sampler,)
        return super()._next_sampler_history(next_sampler)

    def _require_no_evidence_yet(self) -> NoReturn:
        """Raise the standard error for explaining without any evidence."""
        msg = (
            f"no samples yet: sample at least {self.min_budget} evaluations first; "
            "note that sample() returns a new approximator: "
            "`approximator = approximator.sample(budget)`"
        )
        raise InsufficientSamplesError(msg)


def _packed_rows(dense: Array) -> np.ndarray:
    """Return one packed identity row per sample-axis coalition (shared targets only)."""
    rows = np.asarray(dense).reshape(-1, dense.shape[-2], dense.shape[-1])[0]
    return np.packbits(rows, axis=-1)


def _classify_chunk(
    packed: np.ndarray,
    position: int,
    *,
    known: dict[bytes, int],
    batch_ranks: dict[bytes, int],
    novel_positions: list[int],
    state_duplicates: list[tuple[int, int]],
    batch_duplicates: list[tuple[int, int]],
) -> int:
    """Classify one sampled chunk into novel rows and duplicates.

    The classification runs once per distinct coalition of the chunk instead
    of once per row: duplicates within the chunk resolve through
    ``np.unique``, and only the unique rows touch the carried dictionaries.
    Novel ranks are assigned in first-occurrence order, matching sequential
    row-by-row classification exactly.

    Returns:
        The number of novel coalitions found in the chunk.
    """
    unique_rows, first_indices, inverse = np.unique(
        packed,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    codes = np.empty(unique_rows.shape[0], dtype=np.int64)
    is_state_dup = np.zeros(unique_rows.shape[0], dtype=bool)
    found = 0
    for unique_index in np.argsort(first_indices, kind="stable"):
        key = unique_rows[unique_index].tobytes()
        rank = batch_ranks.get(key)
        if rank is not None:
            codes[unique_index] = rank
        elif (source := known.get(key)) is not None:
            is_state_dup[unique_index] = True
            codes[unique_index] = source
        else:
            batch_ranks[key] = len(novel_positions)
            codes[unique_index] = len(novel_positions)
            novel_positions.append(position + int(first_indices[unique_index]))
            found += 1
    row_positions = position + np.arange(packed.shape[0])
    row_codes = codes[inverse]
    row_is_state = is_state_dup[inverse]
    novel_first = np.zeros(packed.shape[0], dtype=bool)
    if found:
        novel_first[np.asarray(novel_positions[-found:]) - position] = True
    state_mask = row_is_state
    batch_mask = ~row_is_state & ~novel_first
    state_duplicates.extend(
        zip(row_positions[state_mask].tolist(), row_codes[state_mask].tolist(), strict=True),
    )
    batch_duplicates.extend(
        zip(row_positions[batch_mask].tolist(), row_codes[batch_mask].tolist(), strict=True),
    )
    return found
