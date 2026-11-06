"""Samplers: draw the coalitions of the next sampling step."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Bool, Float, Int, Key
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.sampling._state import ApproximationState, Sample, SamplingState
from shapiq.typing import typechecked


class Sampler[StateT: ApproximationState](ABC):
    """Draws the sample of the next step and returns its successor.

    The sampler owns the design (sizes, pairing, enumerated coalitions, replacement): it
    records what it knows when drawing in the sample's log weights, and ``log_weights`` turns a
    state into estimator weights. Samplers are immutable: whatever evolves between steps (a
    random key, a permutation cursor, running statistics) lives in the returned sampler, so
    keep using that one.
    """

    @abstractmethod
    def sample(self, state: StateT, budget: int) -> tuple[Sample[Any], Self]:
        """Draw ``budget`` coalitions per entry of the sampler's batch, with their log weights.

        The sampler's batch is the sample's sharing batch: it broadcasts into the game batch,
        so a size-one axis shares its coalitions across those games. Index the sampler to align
        it, e.g. ``sampler[:, None]``.
        """

    def log_weights(self, state: SamplingState[Any]) -> Float[jax.Array, "#*batch n_samples"]:
        """Return weights so that the mean of ``exp(log_weights[i]) * g(S_i)`` estimates totals.

        The mean runs over all samples of ``state`` and estimates the sum of ``g`` over all
        coalitions. Independent steps keep their recorded weights: the mean then counts each
        step's estimate by its share of the samples. Samplers whose steps depend on each other,
        e.g. distinct coalitions across steps, override this.

        Reads only ``state``, so earlier states (``rollback``, ``head``) can be weighted too.
        """
        return state.log_weights


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class RandomSampler[StateT: ApproximationState](Sampler[StateT], JaxAxisProtected[jax.Array], ABC):
    """A sampler that draws coalitions from a random distribution.

    The distribution may depend on the current state, but the next step's coalitions are
    conditionally independent of earlier steps given the current state.

    The batch is the sharing batch, one typed key (``jax.random.key``) per entry: split a key to
    run independent samplers. The size weights stay compact, so one distribution can serve all
    keys.
    """

    keys: Key[jax.Array, "*batch"]
    weights: Float[jax.Array, "#*batch n_sizes"]
    protected_axes: ClassVar[dict[str, int]] = {"keys": 0, "weights": 1}
    broadcasted_batch: ClassVar[set[str]] = {"weights"}

    @property
    def n_players(self) -> int:
        """Return the number of players in the game."""
        return self.weights.shape[-1] - 1

    @classmethod
    def uniform_over_sizes(cls, keys: Key[jax.Array, "*batch"], n_players: int) -> Self:
        """Return a sampler that gives every coalition size the same probability."""
        return cls(keys=keys, weights=jnp.ones((n_players + 1), dtype=float)[None, :])

    @classmethod
    def uniform_over_coalitions(cls, keys: Key[jax.Array, "*batch"], n_players: int) -> Self:
        """Return a sampler that gives every coalition the same probability."""
        log_comb = _log_comb(n_players)[None, :]  # (*batch, n_sizes)
        # Relative to the largest binomial, so the weights stay finite for many players.
        return cls(keys=keys, weights=jnp.exp(log_comb - jnp.max(log_comb)))


def _log_comb(n_players: int) -> Float[jax.Array, " n_sizes"]:
    """Return ``log C(n, s)`` for every size ``s``."""
    size = jnp.arange(n_players + 1)
    return (
        jsp.special.gammaln(n_players + 1.0)
        - jsp.special.gammaln(size + 1.0)
        - jsp.special.gammaln(n_players - size + 1.0)
    )


def _split(keys: Key[jax.Array, "*batch"], num: int) -> tuple[Key[jax.Array, "*batch"], ...]:
    """Split every key of the batch into ``num`` independent keys."""
    parts = jax.vmap(lambda key: jax.random.split(key, num))(keys.reshape(-1))  # (n_keys, num)
    return tuple(parts[:, i].reshape(keys.shape) for i in range(num))


def _uniform(
    keys: Key[jax.Array, "*batch"], shape: tuple[int, ...]
) -> Float[jax.Array, "*batch ..."]:
    """Draw uniforms of ``shape`` per key, independently across the batch."""
    draws = jax.vmap(lambda key: jax.random.uniform(key, shape))(keys.reshape(-1))
    return draws.reshape(*keys.shape, *shape)


def _draw_sizes(
    allocation: Int[jax.Array, "*batch n_sizes"], budget: int
) -> Int[jax.Array, "*batch budget"]:
    """Expand the counts per size into the size of each draw, in ascending order.

    ``budget`` must be the total of every count row; it fixes the output shape under jit.
    """
    counts = allocation.reshape(-1, allocation.shape[-1])  # (batch, n_sizes)
    sizes = jnp.arange(allocation.shape[-1])
    expanded = jax.vmap(lambda row: jnp.repeat(sizes, row, total_repeat_length=budget))(counts)
    return expanded.reshape(*allocation.shape[:-1], budget)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SizeSampler(RandomSampler[SamplingState]):
    """Sample coalitions based on size probabilities, uniform within each size."""

    def _allocate(
        self, key: jax.Array, _state: SamplingState[Any], budget: int
    ) -> Int[jax.Array, "*batch n_sizes"]:
        """Allocate the budget across sizes based on the sampler's weights."""
        weights = jnp.broadcast_to(self.weights, (*self.shape, self.n_players + 1))
        expected_counts = (
            budget * weights / jnp.sum(weights, axis=-1, keepdims=True)
        )  # (*batch, n_sizes)
        segments = (
            jnp.cumsum(expected_counts, axis=-1).at[..., -1].set(budget)
        )  # (*batch, n_sizes); last segment ends at budget
        u = _uniform(key, (1,))  # (*batch, 1); one shift per entry
        below = jnp.clip(
            jnp.ceil(segments - u), 0, budget
        )  # (*batch, n_sizes); how many markers are below each segment
        return jnp.diff(below, axis=-1, prepend=0).astype(
            jnp.int32
        )  # (*batch, n_sizes); how many markers are in each segment

    def _weights(self, sizes: Int[jax.Array, "*batch budget"]) -> Float[jax.Array, "*batch budget"]:
        """Return each draw's log weight ``log C(n, s) - log q(s)``, for its size ``s``.

        A draw of size ``s`` hits a given coalition with probability ``q(s) / C(n, s)``, so
        dividing by it makes the mean over the draws an unbiased estimate of totals. Sizes with
        ``q(s) = 0`` get an infinite weight, but no draws.
        """
        log_q = jnp.log(self.weights) - jnp.log(jnp.sum(self.weights, axis=-1, keepdims=True))
        log_weight_per_size = jnp.broadcast_to(
            _log_comb(self.n_players) - log_q, (*sizes.shape[:-1], self.n_players + 1)
        )  # (*batch, n_sizes)
        return jnp.take_along_axis(log_weight_per_size, sizes, axis=-1)  # (*batch, budget)

    def _draw(
        self, key: jax.Array, sizes: Int[jax.Array, "*batch budget"]
    ) -> Bool[jax.Array, "*batch budget n_players"]:
        """Draw one uniform coalition of each row's size, independently across rows.

        (1) For each potential player, we draw a uniform number in [0, 1).
        (2) Find for each row the s-th smallest number, where s is the row's size.
        (3) Return the players whose numbers are smaller than that threshold.
        """
        u = _uniform(key, (sizes.shape[-1], self.n_players))  # (1) (*batch, budget, n_players)
        # (2) Sort the tickets and append a sentinel value of +inf, so that the s-th smallest ticket is at index s.
        sorted_u = jnp.concatenate(
            [jnp.sort(u, axis=-1), jnp.full((*sizes.shape, 1), jnp.inf)], axis=-1
        )  # (*batch, budget, n_players + 1)
        threshold = jnp.take_along_axis(sorted_u, sizes[..., None], axis=-1)
        # (3) Construct coalitions of exactly size s
        return u < threshold  # (*batch, budget, n_players)

    def sample(self, state: SamplingState[Any], budget: int) -> tuple[Sample[Any], Self]:
        """Draw ``budget`` coalitions per entry of the sampler's batch, with their log weights.

        The mean of ``exp(log_weights[i]) * g(S_i)`` over all samples estimates the sum of
        ``g`` over all coalitions. Independent steps keep their recorded weights: the mean then
        counts each step's estimate by its share of the samples. Samplers whose steps depend on
        each other, e.g. distinct coalitions across steps, override this.
        """
        next_keys, allocate_key, draw_key = _split(self.keys, 3)
        allocation = self._allocate(allocate_key, state, budget)  # Derive counts per size
        sizes = _draw_sizes(allocation, budget)  # The size of each draw, shared by weights and sets
        log_weights = self._weights(
            sizes
        )  # Compute correction log weights for the sampled coalitions
        coalitions = self._draw(draw_key, sizes)  # Draw coalitions based on the allocated sizes
        sample = Sample(
            coalitions=coalitions, log_weights=log_weights, players=state.players, unit=1
        )
        return sample, replace(self, keys=next_keys)
