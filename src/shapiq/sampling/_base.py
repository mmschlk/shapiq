from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from shapiq._shape import Shape, ShapeLike, normalize_shape, validate_int, validate_n_players
from shapiq.coalitions import CoalitionArray, DenseCoalitionArray
from shapiq.sampling._state import ApproximationState

type ShareSamples = bool | int | tuple[int, ...]


class Sampler[StateT: ApproximationState](ABC):
    """Base abstraction for coalition samplers."""

    n_players: int
    target_shape: Shape
    share_samples: ShareSamples
    shared_target_shape: Shape

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        *,
        share_samples: ShareSamples = False,
    ) -> None:
        """Initialize sampler shape policy."""
        self.n_players = validate_n_players(n_players)
        self.target_shape = normalize_shape(target_shape)
        self.share_samples = _validate_share_samples(share_samples, self.target_shape)
        self.shared_target_shape = _shared_target_shape(self.target_shape, self.share_samples)

    @property
    def mutable(self) -> bool:
        """Return whether this sampler mutates in place."""
        return False

    @property
    def sampling_quantum(self) -> int:
        """Return the smallest sample count after which new evidence is usable.

        Estimates incorporate only completed quanta; samples belonging to an
        incomplete quantum stay pending until later sampling completes them.
        Samplers whose evidence is usable per single sample return ``1``.
        """
        return 1

    @property
    def n_seed_samples(self) -> int:
        """Return the number of deterministic seed samples emitted first.

        Seed samples, such as the empty and grand coalition, are emitted as a
        one-time prelude unit before sampled units and are paid from the
        sample budget. Samplers without a seed prelude return ``0``.
        """
        return 0

    @property
    def n_pending_samples(self) -> int:
        """Return the number of sampled coalitions in an incomplete unit.

        Pending samples are already evaluated and stored, but masked from
        explanations until their unit completes on a later sample call.
        """
        return 0

    def sample(self, state: StateT, budget: int) -> tuple[CoalitionArray, Self]:
        """Sample coalitions and return the evolved sampler."""
        validate_int("budget", budget)
        if budget == 0:
            return DenseCoalitionArray.empty(self.sample_shape(0), self.n_players), self
        return self._sample(state, budget)

    def sample_shape(self, budget: int) -> Shape:
        """Return the logical coalition shape for a budget."""
        validate_int("budget", budget)
        return (*self.shared_target_shape, budget)

    @abstractmethod
    def _sample(self, state: StateT, budget: int) -> tuple[CoalitionArray, Self]:
        """Sample positive-budget coalitions."""


def _validate_share_samples(share_samples: ShareSamples, target_shape: Shape) -> ShareSamples:
    if isinstance(share_samples, bool):
        return share_samples
    axes = (share_samples,) if isinstance(share_samples, int) else share_samples
    if not isinstance(axes, tuple):
        msg = (
            "share_samples must be a bool, an int axis, or a tuple of int axes; "
            "samples are not shared by default"
        )
        raise TypeError(msg)
    normalized = tuple(_normalize_axis(axis, len(target_shape)) for axis in axes)
    if len(set(normalized)) != len(normalized):
        msg = "share_samples axes must not contain duplicates"
        raise ValueError(msg)
    return normalized


def _normalize_axis(axis: int, ndim: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        msg = f"share_samples axes must be integers, got {type(axis).__name__}"
        raise TypeError(msg)
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        msg = "share_samples axis out of range"
        raise ValueError(msg)
    return normalized


def _shared_target_shape(target_shape: Shape, share_samples: ShareSamples) -> Shape:
    if share_samples is False or share_samples == ():
        return target_shape
    if share_samples is True:
        return tuple(1 for _ in target_shape)
    axes = (share_samples,) if isinstance(share_samples, int) else share_samples
    return tuple(1 if index in axes else dim for index, dim in enumerate(target_shape))
