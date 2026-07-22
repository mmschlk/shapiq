from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from shapiq._shape import Shape, ShapeLike, normalize_shape, validate_n_players

if TYPE_CHECKING:
    from jax import Array

    from shapiq.coalitions import CoalitionArray

type ShareSamples = bool | int | tuple[int, ...]


@runtime_checkable
class LawfulSampler(Protocol):
    """Optional capability: the marginal law of one sampled coalition.

    Samplers whose draws are identically distributed may declare their law.
    ``log_probability`` answers for the marginal distribution of one drawn
    coalition *after* all wrapper transformations (pairing symmetrizes the
    wrapped law), in log-space so many-player binomials stay finite;
    coalitions outside the support answer ``-inf``. The deterministic seed
    evaluations an approximator makes sit outside the law — they are
    certain, not sampled. Samplers whose draws are not coalitions
    (permutation walks) do not implement the capability:
    Horvitz-Thompson-style estimators check for it, and unit-structured
    estimators never need it.
    """

    def log_probability(self, coalitions: CoalitionArray) -> Array:
        """Return per-coalition log-probabilities under the sampling law."""
        ...


class Sampler(ABC):
    """Base abstraction for draw sources: stateless sampler values.

    A sampler proposes draws — permutations, coalitions, whatever its kind
    stands for — and nothing else: no schedule, no budget, no evolution.
    Draw ``unit_index`` derives from ``fold_in(random_state, unit_index)``,
    so draws are order-free and a sampler never changes: approximators ask
    for the unit indices they need and own everything downstream. Shape
    policy (which explanation targets share draws) is sampler-owned and
    trusted by approximators rather than revalidated per call.
    """

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
        random_state: Array | int = 0,
    ) -> None:
        """Initialize sampler shape policy and the draw key.

        Args:
            n_players: Number of players in the explained game. Must be at
                least two.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            share_samples: Policy for sharing draws across explanation-target
                axes. ``False`` draws independently per target; ``True``
                shares across all target axes; an integer or tuple of
                integers shares across the selected axes.
            random_state: Integer seed or JAX PRNG key from which every draw
                derives.

        Raises:
            ValueError: If ``n_players`` is smaller than two.
            TypeError: If ``random_state`` is neither an integer nor a JAX
                PRNG key.
        """
        self.n_players = validate_n_players(n_players)
        if self.n_players < 2:
            msg = "sampled draws require at least two players"
            raise ValueError(msg)
        self.target_shape = normalize_shape(target_shape)
        self.share_samples = _validate_share_samples(share_samples, self.target_shape)
        self.shared_target_shape = _shared_target_shape(self.target_shape, self.share_samples)
        self._key = _validate_random_state(random_state)

    @property
    def draws_per_unit(self) -> int:
        """Return how many draws one unit index yields (pairing doubles it)."""
        return 1

    @abstractmethod
    def draws(self, unit_indices: Array) -> Array:
        """Return the draws of the given unit indices, stacked on a new leading axis."""


def _validate_random_state(random_state: Array | int) -> Array:
    """Return a JAX PRNG key from an integer seed or an existing key."""
    if isinstance(random_state, bool):
        msg = "random_state must be an integer seed or a JAX PRNG key, got bool"
        raise TypeError(msg)
    if isinstance(random_state, int):
        return jax.random.key(random_state)
    if isinstance(random_state, jax.Array) and jnp.issubdtype(
        random_state.dtype,
        jax.dtypes.prng_key,
    ):
        return random_state
    msg = (
        f"random_state must be an integer seed or a JAX PRNG key, got {type(random_state).__name__}"
    )
    raise TypeError(msg)


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
