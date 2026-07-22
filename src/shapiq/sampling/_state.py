from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import NamedTuple, Protocol, Self, cast

import jax.numpy as jnp
import numpy as np

from shapiq._shape import (
    Shape,
    ShapeLike,
    logical_size,
    normalize_shape,
    validate_int,
)
from shapiq.coalitions import CoalitionArray, DenseCoalitionArray


class UniqueView(NamedTuple):
    """Distinct sampled coalitions of one evidence stream.

    ``coalitions`` holds every distinct coalition once, in first-occurrence
    order; ``first_indices`` locates each one's first stream position,
    ``counts`` its multiplicity, and ``inverse`` maps every stream position
    to its coalition's rank, so indexing the view's coalitions by
    ``inverse`` rebuilds the stream. The index arrays are host NumPy:
    coalition identity is index metadata. Like the stream it summarizes,
    the view is invariant under budget splits.
    """

    coalitions: CoalitionArray
    first_indices: np.ndarray
    counts: np.ndarray
    inverse: np.ndarray


class ApproximationState:
    """Base abstraction for approximator evidence.

    History is identity, not a feature: every state answers ``history()``
    and ``rollback()``, and a state without prior evidence is its own
    single-entry history.
    """

    def rollback(self, steps: int = 1) -> Self:
        """Return a value-equivalent previous state."""
        _validate_history_steps(steps)
        if steps == 0:
            return self
        msg = "cannot roll back past the initial state"
        raise IndexError(msg)

    def history(
        self,
        *,
        reverse: bool = False,
        include_self: bool = True,
    ) -> list[Self]:
        """Return value-equivalent history states."""
        del reverse
        return [self] if include_self else []


@dataclass(frozen=True)
class EmptyState(ApproximationState):
    """Approximation state that holds no evidence yet.

    Approximators are constructed with an empty state so that no game
    evaluation happens before the first sample call; the first sampled
    block replaces it with an evidence-bearing state, where approximation
    history begins.
    """

    @property
    def n_samples(self) -> int:
        """Return the number of sampled coalitions, which is zero."""
        return 0


class SamplingState[ValueT](ApproximationState):  # noqa: PLW1641
    """Approximation state storing sampled coalitions and evaluated values.

    Values are stored in the canonical internal layout — value axes
    leading, then target axes, with the sample axis last — so the sample
    axis is always ``[..., :n]`` regardless of target or value shapes.
    Evidence accumulates as chunks: ``append`` shares the stored chunks and
    adds the new one without copying, and the ``coalitions`` and ``values``
    views concatenate all chunks once, on first access, caching the result.
    Sampling many times between explanations therefore costs one
    concatenation per explanation instead of one per sample call.

    The state also owns coalition identity: ``unique()`` summarizes the
    stream as distinct coalitions with multiplicities and first positions,
    and deduplication consumes the same packed keys — one definition of
    "the same coalition" for every consumer.
    """

    target_shape: Shape

    def __init__(
        self,
        coalitions: CoalitionArray,
        values: ValueT,
        target_shape: ShapeLike = (),
        *,
        _history_cuts: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Initialize the evidence state from one sampled block.

        Args:
            coalitions: The sampled coalitions, as a dense coalition array.
            values: The evaluated values aligned with the coalitions, in the
                canonical layout (value axes leading, sample axis last).
            target_shape: The game's explanation-target shape.
            _history_cuts: Internal compact history checkpoints — one
                ``(n_samples, residual)`` pair per sample call, carried by
                ``append`` and ``rollback``; the residual is opaque
                approximator bookkeeping (the banked budget at the cut).

        Raises:
            TypeError: If the coalitions are not dense.
            ValueError: If the values' sample axis does not pair with the
                coalitions.
        """
        _require_dense(coalitions)
        self.target_shape = normalize_shape(target_shape)
        _validate_values_alignment(coalitions, values)
        self._chunks: tuple[tuple[CoalitionArray, ValueT], ...] = ((coalitions, values),)
        self._packed_keys: np.ndarray | None = None
        self._unique: UniqueView | None = None
        self._history_cuts = (
            ((self.n_samples, 0),) if _history_cuts is None else _history_cuts
        )

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(n_samples={self.n_samples!r}, "
            f"target_shape={self.target_shape!r})"
        )

    @property
    def coalitions(self) -> CoalitionArray:
        """Return all sampled coalitions as one array."""
        return self._materialized()[0]

    @property
    def values(self) -> ValueT:
        """Return all evaluated values, aligned with the coalitions."""
        return self._materialized()[1]

    @property
    def n_samples(self) -> int:
        """Return the number of sampled coalitions."""
        return sum(
            0 if coalitions.shape == () else coalitions.shape[-1]
            for coalitions, _ in self._chunks
        )

    @property
    def shared_target_shape(self) -> Shape:
        """Return the sampler-produced shared target shape."""
        first_coalitions, _ = self._chunks[0]
        if first_coalitions.shape == ():
            return ()
        return first_coalitions.shape[:-1]

    def packed_keys(self) -> np.ndarray:
        """Return one packed-bit identity row per stored coalition.

        Coalition identity is only well-defined when every explanation
        target sees the same coalitions; the keys are computed once per
        state and cached (a value-equivalent swap, like the chunk
        concatenation).

        Returns:
            A host ``uint8`` array of shape ``(n_samples, ceil(n_players / 8))``
            whose rows are equal exactly when the coalitions are.

        Raises:
            ValueError: If the stored coalitions are not shared across
                explanation targets.
        """
        if logical_size(self.shared_target_shape) != 1:
            msg = (
                "coalition identity requires the same coalitions for every "
                "explanation target; sample with share_samples=True (or share "
                "the selected axes)"
            )
            raise ValueError(msg)
        if self._packed_keys is None:
            self._packed_keys = _packed_rows(self.coalitions.to_dense())
        return self._packed_keys

    def unique(self, n_samples: int | None = None) -> UniqueView:
        """Return the distinct stored coalitions with their multiplicities.

        Args:
            n_samples: Number of leading stream samples to view; defaults to
                all stored samples. Estimators pass their usable prefix so
                pending samples of an unfinished unit are excluded.

        Returns:
            A ``UniqueView`` in first-occurrence order. Values at
            ``first_indices`` on the sample axis pair with the view's
            coalitions; ``counts`` are the multiplicities an estimator
            weights by when it solves over distinct coalitions.

        Raises:
            ValueError: If the stored coalitions are not shared across
                explanation targets, or if ``n_samples`` exceeds the stored
                samples.
            TypeError: If ``n_samples`` is not an integer.
        """
        limit = self.n_samples if n_samples is None else n_samples
        if n_samples is not None:
            validate_int("n_samples", n_samples, minimum=0)
            if n_samples > self.n_samples:
                msg = f"the state stores {self.n_samples} samples; cannot view {n_samples}"
                raise ValueError(msg)
        if limit == self.n_samples and self._unique is not None:
            return self._unique
        packed = self.packed_keys()[:limit]
        _, first_seen, inverse_sorted, counts_sorted = np.unique(
            packed,
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        order = np.argsort(first_seen, kind="stable")
        ranks = np.empty(order.shape[0], dtype=np.int64)
        ranks[order] = np.arange(order.shape[0], dtype=np.int64)
        first_indices = first_seen[order]
        view = UniqueView(
            coalitions=self.coalitions[
                (*((slice(None),) * len(self.shared_target_shape)), first_indices)
            ],
            first_indices=first_indices,
            counts=counts_sorted[order],
            inverse=ranks[inverse_sorted],
        )
        if limit == self.n_samples:
            self._unique = view
        return view

    def append(self, coalitions: CoalitionArray, values: ValueT) -> SamplingState[ValueT]:
        """Append sampled coalitions and evaluated values without copying.

        The appended state shares the stored chunks, so a sampling chain
        costs no concatenations until its evidence is read.
        """
        if coalitions.shape == () or coalitions.shape[-1] == 0:
            return self
        if coalitions.n_players != self._chunks[0][0].n_players:
            msg = "coalitions use a different number of players"
            raise ValueError(msg)
        if coalitions.shape[:-1] != self.shared_target_shape:
            msg = "coalitions use a different shared target shape"
            raise ValueError(msg)
        _require_dense(coalitions)
        _validate_values_alignment(coalitions, values, like=self._chunks[0][1])
        appended = copy(self)
        appended._chunks = (*self._chunks, (coalitions, values))  # noqa: SLF001 - evolving a copy of self
        appended._packed_keys = None  # noqa: SLF001 - the caches describe the shorter stream
        appended._unique = None  # noqa: SLF001 - the caches describe the shorter stream
        appended._history_cuts = (  # noqa: SLF001
            *self._history_cuts,
            (appended.n_samples, 0),
        )
        return appended

    def _materialized(self) -> tuple[CoalitionArray, ValueT]:
        """Concatenate the stored chunks once and cache the result.

        The cache swap is value-equivalent, so the state stays immutable in
        every observable sense.
        """
        if len(self._chunks) > 1:
            merged_coalitions = DenseCoalitionArray(
                jnp.concatenate(
                    [jnp.asarray(chunk.to_dense()) for chunk, _ in self._chunks],
                    axis=-2,
                ),
            )
            merged_values = cast(
                "ValueT",
                jnp.concatenate(
                    [jnp.asarray(values) for _, values in self._chunks],
                    axis=-1,
                ),
            )
            self._chunks = ((merged_coalitions, merged_values),)
        return self._chunks[0]

    def rollback(self, steps: int = 1) -> SamplingState[ValueT]:
        """Return a previous sampling state."""
        _validate_history_steps(steps)
        if steps == 0:
            return self
        cuts = self._history_cuts
        if steps >= len(cuts):
            msg = "cannot roll back past the initial state"
            raise IndexError(msg)
        return self._slice_to(cuts[-1 - steps][0])

    def history(
        self,
        *,
        reverse: bool = False,
        include_self: bool = True,
    ) -> list[ApproximationState]:
        """Return value-equivalent sampling-state history."""
        cuts = self._history_cuts
        if not include_self:
            cuts = cuts[:-1]
        states = [self._slice_to(count) for count, _ in cuts]
        if reverse:
            states.reverse()
        return cast("list[ApproximationState]", states)

    def __eq__(self, other: object) -> bool:
        """Compare current sampling evidence for test/debug use."""
        if not isinstance(other, SamplingState):
            return NotImplemented
        return (
            self.target_shape == other.target_shape
            and _dense_equal(self.coalitions, other.coalitions)
            and bool(jnp.array_equal(jnp.asarray(self.values), jnp.asarray(other.values)))
        )

    def _slice_to(self, n_samples: int) -> SamplingState[ValueT]:
        key = (*((slice(None),) * len(self.shared_target_shape)), slice(0, n_samples))
        value_key = (Ellipsis, slice(0, n_samples))
        cuts = tuple(cut for cut in self._history_cuts if cut[0] <= n_samples)
        return type(self)(
            coalitions=self.coalitions[key],
            values=cast("_Indexable", self.values)[value_key],
            target_shape=self.target_shape,
            _history_cuts=cuts,
        )


def _validate_history_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int):
        msg = "steps must be a non-negative integer"
        raise TypeError(msg)
    if steps < 0:
        msg = "steps must be non-negative"
        raise ValueError(msg)


def _require_dense(coalitions: CoalitionArray) -> None:
    if not isinstance(coalitions, DenseCoalitionArray):
        msg = "only dense coalition storage is supported by SamplingState"
        raise TypeError(msg)


def _validate_values_alignment(
    coalitions: CoalitionArray,
    values: object,
    *,
    like: object = None,
) -> None:
    """Reject values whose shape does not pair with the coalitions.

    Deferring this to the lazy chunk concatenation would blame the read
    instead of the append that stored the misaligned block. Values arrive in
    the canonical layout, so the last axis is the sample axis and must match
    the coalitions; with a stored reference block ``like``, the leading axes
    must match it too. Values without a shape (nested sequences) pass
    through and fail at materialization.
    """
    if coalitions.shape == ():
        return
    shape = getattr(values, "shape", None)
    if shape is None:
        return
    n_samples = coalitions.shape[-1]
    like_shape = getattr(like, "shape", None)
    if like_shape is not None:
        expected = (*like_shape[:-1], n_samples)
        if tuple(shape) != expected:
            msg = (
                f"values with shape {tuple(shape)} do not pair with the stored "
                f"evidence: expected {tuple(expected)} (value axes leading, "
                f"then target axes, then {n_samples} samples last)"
            )
            raise ValueError(msg)
        return
    if len(shape) < 1 or shape[-1] != n_samples:
        msg = (
            f"values with shape {tuple(shape)} do not pair with {n_samples} "
            "sampled coalitions: the canonical layout carries value axes "
            "leading, then target axes, with the sample axis last"
        )
        raise ValueError(msg)


def _dense_equal(left: CoalitionArray, right: CoalitionArray) -> bool:
    return left.n_players == right.n_players and bool(
        jnp.array_equal(jnp.asarray(left.to_dense()), jnp.asarray(right.to_dense())),
    )


def coalition_keys(masks: object) -> list[bytes]:
    """Return a hashable identity per sample-axis coalition (shared targets only).

    This is the one definition of coalition identity: the state's packed
    keys and the deduplication policy both derive from it.
    """
    packed = _packed_rows(masks)
    width = packed.shape[-1]
    blob = packed.tobytes()
    return [blob[start : start + width] for start in range(0, len(blob), width)]


def _packed_rows(masks: object) -> np.ndarray:
    """Pack the shared sample-axis coalition rows into bit rows."""
    dense = np.asarray(masks)
    rows = dense.reshape(-1, dense.shape[-2], dense.shape[-1])[0]
    return np.packbits(rows, axis=-1)


class _Indexable(Protocol):
    """Minimal protocol for indexable values."""

    def __getitem__(self, key: object) -> object:
        """Return indexed data."""
