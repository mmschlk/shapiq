from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self, cast

import jax.numpy as jnp

from shapiq._shape import Shape, ensure_bool, normalize_shape
from shapiq.coalitions import CoalitionArray, DenseCoalitionArray
from shapiq.errors import HistoryError


class ApproximationState:
    """Base abstraction for approximator evidence."""

    @property
    def mutable(self) -> bool:
        """Return whether this state mutates in place."""
        return False

    @property
    def track_history(self) -> bool:
        """Return whether history tracking is enabled."""
        return False

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


@dataclass(eq=False, frozen=True)
class SamplingState[ValueT](ApproximationState):  # noqa: PLW1641
    """Approximation state storing sampled coalitions and evaluated values."""

    coalitions: CoalitionArray
    values: ValueT
    target_shape: Shape = ()
    track_history: bool = False
    _history_n_samples: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Normalize metadata and initialize compact history."""
        ensure_bool("track_history", self.track_history)
        target_shape = normalize_shape(self.target_shape)
        object.__setattr__(self, "target_shape", target_shape)
        if self.track_history and self.mutable:
            msg = "mutable states cannot track history"
            raise HistoryError(msg)
        if self.track_history and self._history_n_samples is None:
            object.__setattr__(self, "_history_n_samples", (self.n_samples,))

    @property
    def n_samples(self) -> int:
        """Return the number of sampled coalitions."""
        if self.coalitions.shape == ():
            return 0
        return self.coalitions.shape[-1]

    @property
    def shared_target_shape(self) -> Shape:
        """Return the sampler-produced shared target shape."""
        if self.coalitions.shape == ():
            return ()
        return self.coalitions.shape[:-1]

    def append(self, coalitions: CoalitionArray, values: ValueT) -> SamplingState[ValueT]:
        """Append sampled coalitions and evaluated values."""
        if coalitions.shape == () or coalitions.shape[-1] == 0:
            return self
        if coalitions.n_players != self.coalitions.n_players:
            msg = "coalitions use a different number of players"
            raise ValueError(msg)
        if coalitions.shape[:-1] != self.shared_target_shape:
            msg = "coalitions use a different shared target shape"
            raise ValueError(msg)
        next_coalitions = _append_coalitions(self.coalitions, coalitions)
        next_values = _append_values(self.values, values, axis=len(self.target_shape))
        history = None
        if self.track_history:
            previous = self._history_n_samples or (self.n_samples,)
            history = (*previous, self.n_samples + coalitions.shape[-1])
        return type(self)(
            coalitions=next_coalitions,
            values=next_values,
            target_shape=self.target_shape,
            track_history=self.track_history,
            _history_n_samples=history,
        )

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


def _append_coalitions(left: CoalitionArray, right: CoalitionArray) -> CoalitionArray:
    if isinstance(left, DenseCoalitionArray) and isinstance(right, DenseCoalitionArray):
        return DenseCoalitionArray(
            jnp.concatenate([jnp.asarray(left.to_dense()), jnp.asarray(right.to_dense())], axis=-2),
        )
    msg = "only dense coalition append is supported by SamplingState"
    raise TypeError(msg)


def _append_values(left: object, right: object, *, axis: int) -> object:
    return jnp.concatenate([jnp.asarray(left), jnp.asarray(right)], axis=axis)


def _dense_equal(left: CoalitionArray, right: CoalitionArray) -> bool:
    return left.n_players == right.n_players and bool(
        jnp.array_equal(jnp.asarray(left.to_dense()), jnp.asarray(right.to_dense())),
    )


class _Indexable(Protocol):
    """Minimal protocol for indexable values."""

    def __getitem__(self, key: object) -> object:
        """Return indexed data."""
