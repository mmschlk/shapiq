"""Samples: the coalitions a sampler drew in one step, with their log weights."""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.typing import typechecked


class Sample[OwnerT: JaxLike[Any] = "Sample[Any]"](JaxAxisProtected[OwnerT], ABC):
    """The coalitions a sampler drew in one step, with what its design says about them.

    The sampler sets ``log_weights`` so that the sum of ``exp(log_weights[i]) * g(S_i)``
    estimates the sum of ``g`` over all coalitions, for any ``g``: a weight is one over the
    coalition's expected count, e.g. ``-log(B p(S))`` for ``B`` independent draws from ``p``.
    Weights that change with later steps, as without replacement, hold for every coalition
    drawn so far, and the sum runs over those. Approximators correct their evidence with these
    weights and need nothing else about the design.

    Subclasses choose the storage and expose the dense coalitions; a subclass can also state
    structure among the coalitions, e.g. ``PairedSample``. The batch is the sharing batch: it
    broadcasts into the game batch, so a size-one axis shares the samples across those games.
    """

    @property
    @abstractmethod
    def coalitions(self) -> Bool[jax.Array, "*batch n_samples n_players"]:
        """Return every sampled coalition."""

    @property
    @abstractmethod
    def log_weights(self) -> Float[jax.Array, "*batch n_samples"]:
        """Return the log weight of every sampled coalition."""

    @property
    def n_samples(self) -> int:
        """Return the number of sampled coalitions."""
        return self.log_weights.shape[-1]

    @abstractmethod
    def head(self, n_samples: int) -> Self:
        """Return the first ``n_samples`` samples of this step."""


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class DenseSample(Sample["DenseSample"]):
    """A sample that stores every coalition, without structure among them."""

    # Required slots replace the inherited abstract properties.
    coalitions: Bool[jax.Array, "*batch n_samples n_players"]
    log_weights: Float[jax.Array, "*batch n_samples"]
    protected_axes: ClassVar[dict[str, int]] = {"coalitions": 2, "log_weights": 1}

    @override
    def head(self, n_samples: int) -> Self:
        """Return the first ``n_samples`` samples of this step."""
        return dataclasses.replace(
            self,
            coalitions=self.coalitions[..., :n_samples, :],
            log_weights=self.log_weights[..., :n_samples],
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class PairedSample(Sample["PairedSample"]):
    """A sample of coalitions and their complements, storing one member per pair.

    ``coalitions`` interleaves each anchor with its complement: ``[S_1, ~S_1, S_2, ~S_2, ...]``,
    so the first ``2k`` samples are the first ``k`` pairs. ``log_weights`` holds one weight per
    member, since a coalition and its complement need not share their weight.
    """

    anchors: Bool[jax.Array, "*batch n_pairs n_players"]
    # A required slot replaces the inherited abstract property.
    log_weights: Float[jax.Array, "*batch n_samples"]
    protected_axes: ClassVar[dict[str, int]] = {"anchors": 2, "log_weights": 1}

    def __post_init__(self) -> None:
        """Check that every pair has two log weights.

        Raises:
            ValueError: ``log_weights`` does not hold two weights per anchor.
        """
        if self.log_weights.shape[-1] != 2 * self.n_pairs:
            msg = f"{self.n_pairs} pairs need {2 * self.n_pairs} log weights."
            raise ValueError(msg)

    @property
    def n_pairs(self) -> int:
        """Return the number of sampled pairs."""
        return self.anchors.shape[-2]

    @property
    @override
    def coalitions(self) -> Bool[jax.Array, "*batch n_samples n_players"]:
        """Return every anchor followed by its complement."""
        pairs = jnp.stack([self.anchors, ~self.anchors], axis=-2)  # (*batch, n_pairs, 2, n_players)
        # Interleave the anchors with reshape, as JAX performs row wise flattening.
        return pairs.reshape(*self.anchors.shape[:-2], 2 * self.n_pairs, self.anchors.shape[-1])

    @override
    def head(self, n_samples: int) -> Self:
        """Return the first ``n_samples`` samples of this step, which must be whole pairs.

        Raises:
            ValueError: ``n_samples`` is odd and would split a pair.
        """
        if n_samples % 2:
            msg = f"{n_samples} samples would split a pair."
            raise ValueError(msg)
        return dataclasses.replace(
            self,
            anchors=self.anchors[..., : n_samples // 2, :],
            log_weights=self.log_weights[..., :n_samples],
        )
