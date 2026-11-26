"""Approximators as online estimators, and the histories they keep between steps."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol, Self, cast, override, runtime_checkable

import jax
import jax.numpy as jnp
from flextype import flexdispatch
from jaxtyping import Bool, Float, Int, Shaped
from ndlike.jax_functions import (
    jax_add,
    jax_broadcast_to,
    jax_concatenate,
    jax_expand_dims,
    jax_multiply,
    jax_sum,
)
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.errors import HistoryError, UnsupportedSamplerError
from shapiq.game import FinitePlayerGame, Game
from shapiq.game.exact import discrete_derivative_coefficients
from shapiq.index import DiscreteDerivativeIndex
from shapiq.index._base import Index
from shapiq.sampling._base import Border, Sampler, WithReplacement
from shapiq.typing import typechecked
from shapiq.value import Value
from shapiq.value.value import JaxValue


@dataclass(frozen=True, slots=True, weakref_slot=True)
class History[OwnerT: JaxLike[Any] = "History[Any]"](JaxAxisProtected[OwnerT], ABC):
    """A base class for history objects.

    A history keeps the sampler as it was before every step along a step axis, so rolling back
    restores the sampler that weighs the remaining draws. A new history holds an empty stack
    of the starting sampler; the current sampler lives in the approximator.
    """

    samplers: Shaped[Sampler[Any, Any], "#*batch n_steps"]
    protected_axes: ClassVar[dict[str, int]] = {"samplers": 1}
    broadcasted_batch: ClassVar[set[str]] = {"samplers"}

    @classmethod
    def start(cls, sampler: Sampler[Any, Any]) -> Self:
        """Return an empty history for draws of ``sampler``."""
        return cls(samplers=jax_expand_dims(sampler, -1)[..., :0])

    @property
    def n_steps(self) -> int:
        """Return the number of steps recorded."""
        return self.samplers.shape[-1]

    @property
    def empty(self) -> bool:
        """Return whether nothing has been recorded yet."""
        return self.n_steps == 0

    def _record(self, sampler: Sampler[Any, Any]) -> Shaped[Sampler[Any, Any], "#*batch n_steps"]:
        """Return the samplers with ``sampler``, the state before the new step, appended."""
        return jax_concatenate([self.samplers, jax_expand_dims(sampler, -1)], axis=-1)

    def _check_step(self, step: int) -> None:
        """Check that ``step`` steps can be kept.

        Raises:
            HistoryError: ``step`` is negative or more than the steps recorded.
        """
        if not 0 <= step <= self.n_steps:
            msg = f"Can not keep {step} steps of a history with {self.n_steps} steps."
            raise HistoryError(msg)

    @abstractmethod
    def update(self, sampler: Sampler[Any, Any], *args: Any, **kwargs: Any) -> Self:
        """Return the history updated with a new step, drawn by ``sampler``."""

    @abstractmethod
    def rollback(self, step: int) -> Self:
        """Return the history with only its first ``step`` steps."""


def _append[T: JaxValue](old: T, new: T, axis: int) -> T:
    """Concatenate along the protected ``axis`` (counted from the end) after broadcasting.

    An empty ``old`` is dropped, so the first step sets the value type, e.g. ``VectorValue``.
    """
    batch = jnp.broadcast_shapes(old.shape[: old.ndim + axis], new.shape[: new.ndim + axis])
    trailing = new.shape[new.ndim + axis + 1 :]
    new = jax_broadcast_to(new, (*batch, new.shape[axis], *trailing))
    if old.shape[axis] == 0:
        return new
    old = jax_broadcast_to(old, (*batch, old.shape[axis], *trailing))
    return jax_concatenate([old, new], axis=axis)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class DrawHistory(History["DrawHistory"]):
    """A history of draws from a sampler, with the coalitions and their values.

    ``ends`` counts the draws after every step, the same for the whole batch.
    """

    coalitions: Bool[jax.Array, "#*batch n_draws #n_players"] = field(
        default_factory=lambda: jnp.zeros((0, 1), bool)
    )
    values: Float[JaxValue, "#*batch n_draws"] = field(default_factory=lambda: jnp.zeros((0,)))
    ends: Int[jax.Array, " n_steps"] = field(default_factory=lambda: jnp.zeros((0,), int))
    protected_axes: ClassVar[dict[str, int]] = {"coalitions": 2, "values": 1, "ends": 1}
    broadcasted_batch: ClassVar[set[str]] = {"coalitions", "values", "ends"}

    @override
    def update(
        self,
        sampler: Sampler[Any, Any],
        new_coalitions: Bool[jax.Array, "#*batch n_new_draws n_players"],
        new_values: Float[JaxValue, "#*batch n_new_draws"],
    ) -> Self:
        """Return the history updated with the draws of a new step and their values."""
        coalitions = _append(self.coalitions, new_coalitions, axis=-2)
        values = _append(self.values, new_values, axis=-1)
        ends = jnp.append(self.ends, coalitions.shape[-2])
        samplers = self._record(sampler)
        return replace(self, samplers=samplers, coalitions=coalitions, values=values, ends=ends)

    @override
    def rollback(self, step: int) -> Self:
        """Return the history with only the draws of its first ``step`` steps."""
        self._check_step(step)
        end = int(self.ends[step - 1]) if step else 0
        return replace(
            self,
            samplers=self.samplers[..., :step],
            coalitions=self.coalitions[..., :end, :],
            values=self.values[..., :end],
            ends=self.ends[:step],
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class EstimateHistory(History["EstimateHistory"]):
    """A history of estimates from an approximator, with the values and their weights.

    What ``weights`` holds is up to the approximator: what it needs to update the latest
    estimate when later steps add draws.
    """

    estimates: Float[JaxValue, "#*batch n_steps #n_interactions"] = field(
        default_factory=lambda: jnp.zeros((0, 1))
    )
    weights: Float[jax.Array, "#*batch n_steps"] = field(default_factory=lambda: jnp.zeros((0,)))
    protected_axes: ClassVar[dict[str, int]] = {"estimates": 2, "weights": 1}
    broadcasted_batch: ClassVar[set[str]] = {"estimates", "weights"}

    @override
    def update(
        self,
        sampler: Sampler[Any, Any],
        new_estimates: Float[JaxValue, "#*batch n_new_steps n_interactions"],
        new_weights: Float[jax.Array, "#*batch n_new_steps"],
    ) -> Self:
        """Return the history updated with the estimate of a new step and its weight."""
        estimates = _append(self.estimates, new_estimates, axis=-2)
        weights = _append(self.weights, new_weights, axis=-1)
        samplers = self._record(sampler)
        return replace(self, samplers=samplers, estimates=estimates, weights=weights)

    @override
    def rollback(self, step: int) -> Self:
        """Return the history with only the estimates of its first ``step`` steps."""
        self._check_step(step)
        return replace(
            self,
            samplers=self.samplers[..., :step],
            estimates=self.estimates[..., :step, :],
            weights=self.weights[..., :step],
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Approximator[P, V: Value, R: Game, SamplerT: Sampler[Any, Any], IndexT: Index](
    JaxAxisProtected["Approximator"], ABC
):
    """Base class for approximators.

    A new approximator starts the history that ``start_history`` picks for its sampler, and a
    call goes through ``compute``, which dispatches on the type of that history. The draws,
    ``_compute_with_draws``, keep every coalition and value and reweigh them with the latest
    ``sampler.log_weights``, so they work with any sampler; faster paths are optional methods,
    e.g. ``_compute_with_estimates`` (see ``SupportsEstimates``).
    """

    sampler: Shaped[SamplerT, "#*batch"]
    game: FinitePlayerGame[P, V]
    index: Shaped[IndexT, "#*batch"]
    history: Shaped[History[Any], "#*batch"] | None = None
    protected_axes: ClassVar[dict[str, int]] = {"game": 0, "sampler": 0, "index": 0, "history": 0}
    broadcasted_batch: ClassVar[set[str]] = {"sampler", "index", "game", "history"}

    def __post_init__(self) -> None:
        """Start the history unless one is passed along, e.g. by ``replace`` or slicing."""
        if self.history is None:
            object.__setattr__(self, "history", start_history(self.sampler, self))

    def __call__(self, budget: int) -> tuple[R, Self]:
        """Draw ``budget`` more coalitions; return the estimate and the updated approximator."""
        return compute(cast("History[Any]", self.history), self, budget)

    def rollback(self, step: int) -> Self:
        """Return the approximator as it was after its first ``step`` steps.

        The sampler goes back to its state before step ``step``, so the remaining draws keep
        their weights and later calls draw as if the dropped steps never happened.

        Raises:
            HistoryError: ``step`` is negative or more than the steps recorded.
        """
        history = cast("History[Any]", self.history)
        history._check_step(step)  # noqa: SLF001
        if step == history.n_steps:
            return self
        return replace(self, sampler=history.samplers[..., step], history=history.rollback(step))

    @abstractmethod
    def _compute_with_draws(self, budget: int) -> tuple[R, Self]:
        """Add ``budget`` draws to all draws so far and estimate from all of them.

        Every approximator implements it: every sampler weighs its draws with ``log_weights``,
        so the draws work with any sampler.
        """


@runtime_checkable
class SupportsEstimates(Protocol):
    """An approximator that updates its latest estimate, for samplers that rescale all weights.

    An approximator supports it by having the method; it needs no base class.
    """

    def _compute_with_estimates(self, budget: int) -> tuple[Any, Any]:
        """Update the latest estimate with ``budget`` new draws."""
        ...


# What ``start_history`` returns, ``compute`` has to handle: keep the two side by side.


@flexdispatch
def start_history(
    sampler: Sampler[Any, Any], approximator: Approximator[Any, Any, Any, Any, Any]
) -> History[Any]:
    """Return the empty history ``approximator`` keeps for draws of ``sampler``.

    The draws follow any sampler.
    """
    del approximator
    return DrawHistory.start(sampler)


@start_history.register(WithReplacement)
def _(
    sampler: WithReplacement[Any, Any], approximator: Approximator[Any, Any, Any, Any, Any]
) -> History[Any]:
    """Every earlier weight scales by the same factor, so the latest estimate is enough."""
    if isinstance(approximator, SupportsEstimates):
        return EstimateHistory.start(sampler)
    return DrawHistory.start(sampler)


@start_history.register(Border)
def _(
    sampler: Border[Any, Any], approximator: Approximator[Any, Any, Any, Any, Any]
) -> History[Any]:
    """Enumerated strata reweigh their draws apart from the rest, so only the draws follow."""
    del approximator
    return DrawHistory.start(sampler)


@flexdispatch
def compute(
    history: History[Any], approximator: Approximator[Any, Any, Any, Any, Any], budget: int
) -> tuple[Any, Any]:
    """Draw ``budget`` more coalitions with the computation that fits ``history``."""
    del history
    return approximator._compute_with_draws(budget)  # noqa: SLF001


@compute.register(EstimateHistory)
def _(history: EstimateHistory, approximator: SupportsEstimates, budget: int) -> tuple[Any, Any]:
    """Update the latest estimate."""
    del history
    return approximator._compute_with_estimates(budget)  # noqa: SLF001


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SHAPIQ[
    P,
    V: Float[JaxValue, "..."],
    R: Game,
    SamplerT: Sampler[Any, Any],
    IndexT: DiscreteDerivativeIndex,
](Approximator[P, V, R, SamplerT, IndexT]):
    """SHAPIQ approximator: the weighted sum of every draw's discrete derivatives.

    With the estimate history, ``weights`` holds the sampler's expected units so far, to which
    every earlier draw's weight is inversely proportional.
    """

    def _weighted_sum(
        self,
        coalitions: Bool[jax.Array, "#*batch n_draws n_players"],
        values: Float[JaxValue, "#*batch n_draws"],
        log_weights: Float[jax.Array, "#*batch n_draws"],
    ) -> Float[JaxValue, "*batch n_interactions"]:
        """Return the sum over draws of weight times value times derivative coefficients.

        The ndlike functions let vector values, e.g. ``VectorValue``, carry their outputs along.
        """
        interactions = self.index.interactions(self.game.n_players)
        coefficients = discrete_derivative_coefficients(self.index, coalitions, interactions)
        weighted = jax_multiply(values, jnp.exp(log_weights))  # (*batch, n_draws)
        return jax_sum(jax_multiply(coefficients, jax_expand_dims(weighted, -2)), axis=-1)

    def _build(self, estimate: Float[JaxValue, "*batch n_interactions"]) -> R:
        """Return the estimate as a game of the index."""
        interactions = self.index.interactions(self.game.n_players)
        return self.index.build(estimate, interactions, self.game.players)

    def _compute_with_estimates(self, budget: int) -> tuple[R, Self]:
        """Rescale the latest estimate to the new number of units and add the new draws.

        Raises:
            UnsupportedSamplerError: The sampler is not with replacement.
        """
        if not isinstance(self.sampler, WithReplacement):
            msg = (
                f"The estimate history needs a WithReplacement, not {type(self.sampler).__name__}."
            )
            raise UnsupportedSamplerError(msg)
        history = cast("EstimateHistory", self.history)
        sample, sampler = self.sampler.sample(budget)
        # A trailing size-one game axis lets the sample axis broadcast like a batch axis.
        values = self.game[..., None](sample.coalitions)  # (*batch, B)
        estimate = self._weighted_sum(sample.coalitions, values, sample.log_weights)
        units = jnp.sum(sampler.expected, axis=-1)  # (*batch)
        if not history.empty:
            # Earlier weights are 1 / (units p(S)): they shrink by the ratio of the units.
            ratio = history.weights[..., -1] / units
            estimate = jax_add(
                estimate, jax_multiply(ratio[..., None], history.estimates[..., -1, :])
            )
        history = history.update(self.sampler, estimate[..., None, :], units[..., None])
        return self._build(estimate), replace(self, sampler=sampler, history=history)

    @override
    def _compute_with_draws(self, budget: int) -> tuple[R, Self]:
        """Add the new draws to all draws so far and reweigh all of them with the new sampler."""
        history = cast("DrawHistory", self.history)
        sample, sampler = self.sampler.sample(budget)
        values = self.game[..., None](sample.coalitions)  # (*batch, B)
        history = history.update(self.sampler, sample.coalitions, values)
        log_weights = sampler.log_weights(history.coalitions)
        estimate = self._weighted_sum(history.coalitions, history.values, log_weights)
        return self._build(estimate), replace(self, sampler=sampler, history=history)


## TODO(advueu963): Restructure the function to allow for jit compilation. currently the Jit recompile due to the history changes.  # noqa: TD003
## Compilation could be done based on the budget, as this determines the shapes. Then hot-paths must use the same budgets across each run.
