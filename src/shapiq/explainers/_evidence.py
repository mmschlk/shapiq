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
        """Return the smallest total budget after which ``explain()`` works.

        The first explanation needs the seed block plus one completed sampled
        unit; this property saves users the arithmetic over the sampler's
        ``n_seed_samples`` and ``sampling_quantum``.
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
            found = 0
            for key in _coalition_keys(dense):
                if key in batch_ranks:
                    batch_duplicates.append((position, batch_ranks[key]))
                elif key in known:
                    state_duplicates.append((position, known[key]))
                else:
                    batch_ranks[key] = len(novel_positions)
                    novel_positions.append(position)
                    found += 1
                position += 1
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
        dense = jnp.asarray(self.state.coalitions.to_dense())
        known: dict[bytes, int] = {}
        for index, key in enumerate(_coalition_keys(dense)):
            known.setdefault(key, index)
        return known

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


def _coalition_keys(dense: Array) -> list[bytes]:
    """Return a hashable identity per sample-axis coalition (shared targets only)."""
    rows = np.asarray(dense).reshape(-1, dense.shape[-2], dense.shape[-1])[0]
    packed = np.packbits(rows, axis=-1)
    width = packed.shape[-1]
    blob = packed.tobytes()
    return [blob[start : start + width] for start in range(0, len(blob), width)]
