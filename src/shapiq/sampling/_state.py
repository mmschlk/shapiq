from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Protocol, Self, cast

import jax.numpy as jnp

from shapiq._shape import Shape, ShapeLike, ensure_bool, normalize_shape
from shapiq.coalitions import CoalitionArray, DenseCoalitionArray
from shapiq.errors import HistoryError


class ApproximationState:
    """Base abstraction for approximator evidence.

    ``mutable`` declares value-visible in-place mutation; value-equivalent
    internal caching does not count. Both flags are plain class attributes
    so subclasses may shadow them with fields or instance assignments.
    """

    mutable: bool = False
    track_history: bool = False

    def rollback(self, steps: int = 1) -> Self:
        """Return a value-equivalent previous state."""
        _validate_history_steps(steps)
        msg = "history is not enabled; construct the approximator with track_history=True"
        raise HistoryError(msg)

    def history(
        self,
        *,
        reverse: bool = False,
        include_self: bool = True,
    ) -> list[Self]:
        """Return value-equivalent history states."""
        ensure_bool("reverse", reverse)
        ensure_bool("include_self", include_self)
        msg = "history is not enabled; construct the approximator with track_history=True"
        raise HistoryError(msg)


@dataclass(frozen=True)
class EmptyState(ApproximationState):
    """Approximation state that holds no evidence yet.

    Approximators are constructed with an empty state so that no game
    evaluation happens before the first sample call. The first sampled batch
    replaces it with an evidence-bearing state, carrying over the history
    preference. Approximation history begins at that first evidence state, so
    an empty state with history enabled lists only itself.
    """

    track_history: bool = False

    def __post_init__(self) -> None:
        """Validate the history preference."""
        ensure_bool("track_history", self.track_history)

    @property
    def n_samples(self) -> int:
        """Return the number of sampled coalitions, which is zero."""
        return 0

    def rollback(self, steps: int = 1) -> Self:
        """Return this state for zero steps; nothing precedes an empty state.

        Args:
            steps: Number of history steps to roll back. Only ``0`` is valid
                for an empty state.

        Returns:
            This state when ``steps`` is ``0``.

        Raises:
            HistoryError: If history is not enabled, if ``steps`` is not a
                non-negative integer, or if ``steps`` is positive.
        """
        _validate_history_steps(steps)
        if not self.track_history:
            msg = "history is not enabled; construct the approximator with track_history=True"
            raise HistoryError(msg)
        if steps == 0:
            return self
        msg = "cannot roll back past the initial state"
        raise HistoryError(msg)

    def history(
        self,
        *,
        reverse: bool = False,
        include_self: bool = True,
    ) -> list[ApproximationState]:
        """Return the single-entry history of an empty state.

        Args:
            reverse: Whether to order history from newest to oldest.
            include_self: Whether to include this state itself.

        Returns:
            A list containing only this state, or an empty list when
            ``include_self`` is false.

        Raises:
            HistoryError: If history is not enabled.
        """
        ensure_bool("reverse", reverse)
        ensure_bool("include_self", include_self)
        if not self.track_history:
            msg = "history is not enabled; construct the approximator with track_history=True"
            raise HistoryError(msg)
        return [self] if include_self else []


class SamplingState[ValueT](ApproximationState):  # noqa: PLW1641
    """Approximation state storing sampled coalitions and evaluated values.

    Evidence accumulates as chunks: ``append`` shares the stored chunks and
    adds the new one without copying, and the ``coalitions`` and ``values``
    views concatenate all chunks once, on first access, caching the result.
    Sampling many times between explanations therefore costs one
    concatenation per explanation instead of one per sample call.
    """

    target_shape: Shape
    track_history: bool

    def __init__(
        self,
        coalitions: CoalitionArray,
        values: ValueT,
        target_shape: ShapeLike = (),
        *,
        track_history: bool = False,
        _history_n_samples: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize the evidence state from one sampled block.

        Args:
            coalitions: The sampled coalitions, as a dense coalition array.
            values: The evaluated values aligned with the coalitions.
            target_shape: The game's explanation-target shape.
            track_history: Whether to record value-equivalent history.
            _history_n_samples: Internal compact history carried by
                ``append`` and ``rollback``.

        Raises:
            TypeError: If the coalitions are not dense.
            ValueError: If the values' sample axis does not pair with the
                coalitions.
        """
        ensure_bool("track_history", track_history)
        _require_dense(coalitions)
        self.target_shape = normalize_shape(target_shape)
        self.track_history = track_history
        _validate_values_alignment(coalitions, values, axis=len(self.target_shape))
        self._chunks: tuple[tuple[CoalitionArray, ValueT], ...] = ((coalitions, values),)
        self._history_n_samples = _history_n_samples
        if track_history and self.mutable:
            msg = "mutable states cannot track history"
            raise HistoryError(msg)
        if track_history and _history_n_samples is None:
            self._history_n_samples = (self.n_samples,)

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(n_samples={self.n_samples!r}, "
            f"target_shape={self.target_shape!r}, track_history={self.track_history!r})"
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
        _validate_values_alignment(
            coalitions,
            values,
            axis=len(self.target_shape),
            like=self._chunks[0][1],
        )
        appended = copy(self)
        appended._chunks = (*self._chunks, (coalitions, values))  # noqa: SLF001 - evolving a copy of self
        if self.track_history:
            previous = self._history_n_samples or (self.n_samples,)
            appended._history_n_samples = (*previous, appended.n_samples)  # noqa: SLF001
        return appended

    def _materialized(self) -> tuple[CoalitionArray, ValueT]:
        """Concatenate the stored chunks once and cache the result.

        The cache swap is value-equivalent, so the state stays immutable in
        the sense ``mutable`` declares.
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
                    axis=len(self.target_shape),
                ),
            )
            self._chunks = ((merged_coalitions, merged_values),)
        return self._chunks[0]

    def rollback(self, steps: int = 1) -> SamplingState[ValueT]:
        """Return a previous sampling state."""
        _validate_history_steps(steps)
        if steps == 0:
            return self
        history = self._require_history()
        if steps >= len(history):
            msg = "cannot roll back past the initial state"
            raise HistoryError(msg)
        return self._slice_to(history[-1 - steps])

    def history(
        self,
        *,
        reverse: bool = False,
        include_self: bool = True,
    ) -> list[ApproximationState]:
        """Return value-equivalent sampling-state history."""
        ensure_bool("reverse", reverse)
        ensure_bool("include_self", include_self)
        counts = self._require_history()
        if not include_self:
            counts = counts[:-1]
        states = [self._slice_to(count) for count in counts]
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

    def _require_history(self) -> tuple[int, ...]:
        if not self.track_history or self._history_n_samples is None:
            msg = "history is not enabled; construct the approximator with track_history=True"
            raise HistoryError(msg)
        return self._history_n_samples

    def _slice_to(self, n_samples: int) -> SamplingState[ValueT]:
        key = (*((slice(None),) * len(self.shared_target_shape)), slice(0, n_samples))
        value_key = (*((slice(None),) * len(self.target_shape)), slice(0, n_samples), Ellipsis)
        history = tuple(count for count in self._require_history() if count <= n_samples)
        return type(self)(
            coalitions=self.coalitions[key],
            values=cast("_Indexable", self.values)[value_key],
            target_shape=self.target_shape,
            track_history=True,
            _history_n_samples=history,
        )


def _validate_history_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int):
        msg = "steps must be a non-negative integer"
        raise HistoryError(msg)
    if steps < 0:
        msg = "steps must be non-negative"
        raise HistoryError(msg)


def _require_dense(coalitions: CoalitionArray) -> None:
    if not isinstance(coalitions, DenseCoalitionArray):
        msg = "only dense coalition storage is supported by SamplingState"
        raise TypeError(msg)


def _validate_values_alignment(
    coalitions: CoalitionArray,
    values: object,
    *,
    axis: int,
    like: object = None,
) -> None:
    """Reject values whose shape does not pair with the coalitions.

    Deferring this to the lazy chunk concatenation would blame the read
    instead of the append that stored the misaligned block. The sample axis
    must match the coalitions; with a stored reference block ``like``, the
    remaining axes must match it too. Values without a shape (nested
    sequences) pass through and fail at materialization.
    """
    if coalitions.shape == ():
        return
    shape = getattr(values, "shape", None)
    if shape is None:
        return
    n_samples = coalitions.shape[-1]
    like_shape = getattr(like, "shape", None)
    if like_shape is not None and len(like_shape) > axis:
        expected = (*like_shape[:axis], n_samples, *like_shape[axis + 1 :])
        if tuple(shape) != expected:
            msg = (
                f"values with shape {tuple(shape)} do not pair with the stored "
                f"evidence: expected {tuple(expected)} (target axes, then "
                f"{n_samples} samples, then value axes)"
            )
            raise ValueError(msg)
        return
    if len(shape) <= axis or shape[axis] != n_samples:
        msg = (
            f"values with shape {tuple(shape)} do not pair with {n_samples} "
            f"sampled coalitions: the sample axis follows the target axes "
            f"(axis {axis}), then any value axes trail"
        )
        raise ValueError(msg)


def _dense_equal(left: CoalitionArray, right: CoalitionArray) -> bool:
    return left.n_players == right.n_players and bool(
        jnp.array_equal(jnp.asarray(left.to_dense()), jnp.asarray(right.to_dense())),
    )


class _Indexable(Protocol):
    """Minimal protocol for indexable values."""

    def __getitem__(self, key: object) -> object:
        """Return indexed data."""
