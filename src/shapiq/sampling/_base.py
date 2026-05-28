from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from shapiq._shape import Shape, ShapeLike, normalize_shape, validate_int, validate_n_players
from shapiq.coalitions import CoalitionArray, DenseCoalitionArray
from shapiq.sampling._state import ApproximationState

type SampleSharing = None | bool | int | tuple[int, ...]


class Sampler[StateT: ApproximationState](ABC):
    """Base abstraction for coalition samplers."""

    n_players: int
    target_shape: Shape
    sample_sharing: SampleSharing
    shared_target_shape: Shape

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        sample_sharing: SampleSharing = None,
    ) -> None:
        """Initialize sampler shape policy."""
        self.n_players = validate_n_players(n_players)
        self.target_shape = normalize_shape(target_shape)
        self.sample_sharing = _validate_sample_sharing(sample_sharing, self.target_shape)
        self.shared_target_shape = _shared_target_shape(self.target_shape, self.sample_sharing)

    @property
    def mutable(self) -> bool:
        """Return whether this sampler mutates in place."""
        return False

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


def _validate_sample_sharing(sample_sharing: SampleSharing, target_shape: Shape) -> SampleSharing:
    if sample_sharing is False:
        msg = "sample_sharing=False is ambiguous; use None"
        raise ValueError(msg)
    if sample_sharing is None or sample_sharing is True:
        return sample_sharing
    axes = (sample_sharing,) if isinstance(sample_sharing, int) else sample_sharing
    if not isinstance(axes, tuple):
        msg = "sample_sharing must be None, True, int, or tuple[int, ...]"
        raise TypeError(msg)
    normalized = tuple(_normalize_axis(axis, len(target_shape)) for axis in axes)
    if len(set(normalized)) != len(normalized):
        msg = "sample_sharing axes must not contain duplicates"
        raise ValueError(msg)
    return normalized


def _normalize_axis(axis: int, ndim: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        msg = "sample_sharing axes must be integers, not bools"
        raise TypeError(msg)
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        msg = "sample_sharing axis out of range"
        raise ValueError(msg)
    return normalized


def _shared_target_shape(target_shape: Shape, sample_sharing: SampleSharing) -> Shape:
    if sample_sharing is None or sample_sharing == ():
        return target_shape
    if sample_sharing is True:
        return tuple(1 for _ in target_shape)
    axes = (sample_sharing,) if isinstance(sample_sharing, int) else sample_sharing
    return tuple(1 if index in axes else dim for index, dim in enumerate(target_shape))
