"""Sampling evidence: the samples of each step and the values the games gave them."""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import accumulate
from typing import ClassVar, Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Shaped
from ndlike.jax_functions import jax_concatenate
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.errors import HistoryError, SamplingError
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue


@typechecked
class ApproximationState(ABC):
    """Evidence an approximator collects, with value-equivalent earlier states."""

    @abstractmethod
    def rollback(self, steps: int = 1) -> Self:
        """Return the state from ``steps`` steps ago."""

    @abstractmethod
    def history(self, *, reverse: bool = False, include_self: bool = True) -> list[Self]:
        """Return the state after each step, oldest first unless ``reverse``."""


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class Sample[P](JaxAxisProtected[jax.Array]):
    """The coalitions a sampler drew in one step, with what its design says about them.

    The sampler sets ``log_weights`` so that the mean of ``exp(log_weights[i]) * g(S_i)``
    estimates the sum of ``g`` over all coalitions, for any ``g``; e.g. ``-log p(S)`` for
    independent draws from ``p``. Approximators correct their evidence with these weights and
    need nothing else about the design. Consecutive blocks of ``unit`` samples were drawn
    together, e.g. a coalition and its complement.

    The batch is the sharing batch: it broadcasts into the game batch, so a size-one axis
    shares the samples across those games.
    """

    coalitions: Bool[jax.Array, "#*batch n_samples {players.n_players}"]
    log_weights: Float[jax.Array, "#*batch n_samples"]
    players: Shaped[FinitePlayers[P], "#*batch"]
    unit: int
    protected_axes: ClassVar[dict[str, int]] = {"coalitions": 2, "log_weights": 1, "players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players"}

    def __post_init__(self) -> None:
        """Check that the samples split into whole units.

        Raises:
            ValueError: ``unit`` is not positive or does not divide the number of samples.
        """
        if self.unit < 1 or self.n_samples % self.unit:
            msg = f"{self.n_samples} samples do not split into units of {self.unit}."
            raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Return the number of sampled coalitions."""
        return self.coalitions.shape[-2]

    def head(self, n_samples: int) -> Self:
        """Return the first ``n_samples`` samples of this step."""
        return dataclasses.replace(
            self,
            coalitions=self.coalitions[..., :n_samples, :],
            log_weights=self.log_weights[..., :n_samples],
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class Evaluations[P](Sample[P]):
    """A sample and the values the games gave its coalitions.

    The batch is the game batch of ``values``. The sample's fields stay compact: a sharing
    axis of size one shares its samples across those games.
    """

    values: Shaped[JaxValue, "*batch n_samples"]
    protected_axes: ClassVar[dict[str, int]] = {"values": 1}
    broadcasted_batch: ClassVar[set[str]] = {"coalitions", "log_weights", "players"}

    @classmethod
    def from_sample(cls, sample: Sample[P], values: Shaped[JaxValue, "*batch n_samples"]) -> Self:
        """Attach the values the games gave the coalitions of ``sample``."""
        return cls(
            coalitions=sample.coalitions,
            log_weights=sample.log_weights,
            players=sample.players,
            unit=sample.unit,
            values=values,
        )

    @override
    def head(self, n_samples: int) -> Self:
        """Return the first ``n_samples`` samples of this step."""
        return dataclasses.replace(
            self,
            coalitions=self.coalitions[..., :n_samples, :],
            log_weights=self.log_weights[..., :n_samples],
            values=self.values[..., :n_samples],
        )


def _concatenate_shared(parts: list[jax.Array], n_internal: int) -> jax.Array:
    """Broadcast compact parts to their common shared batch, then join them along the samples.

    ``n_internal`` counts the trailing axes after the batch; the first of them is the sample axis.
    """
    shared = jnp.broadcast_shapes(*(part.shape[: part.ndim - n_internal] for part in parts))
    parts = [
        jnp.broadcast_to(part, (*shared, *part.shape[part.ndim - n_internal :])) for part in parts
    ]
    return jnp.concatenate(parts, axis=-n_internal)  # (*shared, n_samples, ...)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SamplingState[P](ApproximationState):
    """Sampling evidence as the ordered steps that collected it.

    ``append`` returns a new state sharing the earlier steps, so every earlier state is a
    prefix and history needs no extra bookkeeping. The first step fixes the game batch.
    """

    players: FinitePlayers[P]
    steps: tuple[Evaluations[P], ...] = ()

    def __post_init__(self) -> None:
        """Check that all steps share the players, player count, and game batch.

        Raises:
            ValueError: A step uses other players or another game batch.
        """
        for step in self.steps:
            if step.players is not self.players:
                msg = "Every sampling step must use the state's players."
                raise ValueError(msg)
            if step.shape != self.steps[0].shape:
                msg = "Every sampling step must cover the same game batch."
                raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Return the number of sampled coalitions."""
        return sum(step.n_samples for step in self.steps)

    @property
    def history_n_samples(self) -> tuple[int, ...]:
        """Return the cumulative number of samples after each step."""
        return tuple(accumulate(step.n_samples for step in self.steps))

    @property
    def values(self) -> Shaped[JaxValue, "*batch n_samples"]:
        """Return the values of all steps, concatenated along the sample axis.

        Raises:
            SamplingError: The state holds no samples yet.
        """
        self._require_samples()
        return jax_concatenate([step.values for step in self.steps], axis=-1)

    @property
    def coalitions(self) -> Bool[jax.Array, "#*batch n_samples n_players"]:
        """Return the coalitions of all steps, broadcast to a common shared batch.

        Raises:
            SamplingError: The state holds no samples yet.
        """
        self._require_samples()
        return _concatenate_shared([step.coalitions for step in self.steps], n_internal=2)

    @property
    def log_weights(self) -> Float[jax.Array, "#*batch n_samples"]:
        """Return the log weights the steps' samplers recorded, concatenated along the samples.

        These are the draw-time records. How they combine into estimator weights is the
        sampler's decision: see ``Sampler.log_weights``.

        Raises:
            SamplingError: The state holds no samples yet.
        """
        self._require_samples()
        return _concatenate_shared([step.log_weights for step in self.steps], n_internal=1)

    def append(self, sample: Sample[P], values: Shaped[JaxValue, "*batch n_samples"]) -> Self:
        """Return the state with one more step: ``sample`` and the values the games gave it."""
        if sample.n_samples == 0:
            return self
        step = Evaluations.from_sample(sample, values)
        return dataclasses.replace(self, steps=(*self.steps, step))

    def head(self, n_samples: int) -> Self:
        """Return the state holding the first ``n_samples`` samples, cutting into a step if needed.

        Raises:
            HistoryError: ``n_samples`` is negative or exceeds the collected samples.
        """
        if not 0 <= n_samples <= self.n_samples:
            msg = "n_samples must lie between 0 and the number of collected samples."
            raise HistoryError(msg)
        steps, remaining = [], n_samples
        for step in self.steps:
            if remaining == 0:
                break
            take = min(remaining, step.n_samples)
            steps.append(step if take == step.n_samples else step.head(take))
            remaining -= take
        return dataclasses.replace(self, steps=tuple(steps))

    def rollback(self, steps: int = 1) -> Self:
        """Return the state from ``steps`` steps ago; rolling back every step leaves it empty.

        Raises:
            HistoryError: ``steps`` is negative or exceeds the number of steps.
        """
        if not 0 <= steps <= len(self.steps):
            msg = "steps must lie between 0 and the number of sampling steps."
            raise HistoryError(msg)
        return dataclasses.replace(self, steps=self.steps[: len(self.steps) - steps])

    def history(self, *, reverse: bool = False, include_self: bool = True) -> list[Self]:
        """Return the state after each step, oldest first unless ``reverse``."""
        n_steps = len(self.steps) if include_self else len(self.steps) - 1
        states = [dataclasses.replace(self, steps=self.steps[:i]) for i in range(1, n_steps + 1)]
        return states[::-1] if reverse else states

    def _require_samples(self) -> None:
        if not self.steps:
            msg = "The state holds no samples yet."
            raise SamplingError(msg)
