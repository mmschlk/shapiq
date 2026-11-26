"""Samplers: draw the coalitions of the next sampling step.

A sampler is a class listing its parts, in this order:

1. modifiers that adjust hooks: ``Paired``, ``Border``;
2. a design that fills the stratum hooks: ``Uniform``, ``Size``;
3. a mechanism that runs the step: ``WithReplacement``, ``WithoutReplacement``.

For example ``class PairedBorderSizeWithReplacement(Paired, Border, Size, WithReplacement)``.
The base order is the hook chain: a modifier's ``super()`` reaches the design, then the
mechanism, then the defaults on ``Sampler``. ``WithoutReplacement`` enumerates full strata by
itself, so it is already a ``Border``.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from math import comb
from typing import Any, ClassVar, Self, cast, override

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jaxtyping import Bool, Float, Int, Key, UInt32
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.errors import SamplingError
from shapiq.sampling._permutation import (
    add_mod,
    bits,
    colex_unrank,
    from_int,
    from_positions,
    n_rounds,
    n_words,
    random_below_each,
    swap_or_not,
)
from shapiq.sampling._state import DenseSample, PairedSample, Sample
from shapiq.typing import typechecked

# Classes without fields are plain classes with empty slots. Designs carry fields and are
# combined with either mechanism, whose fields are slots too; two bases with slotted fields
# can not be combined, so the designs keep a ``__dict__``.


class Sampler[SampleT: Sample[Any] = DenseSample, OwnerT: JaxLike[Any] = "Sampler[Any, Any]"](
    JaxAxisProtected[OwnerT], ABC
):
    """Draws the sample of the next step and returns its successor.

    Samplers are immutable: whatever evolves between steps (a random key, a permutation
    cursor, running statistics) lives in the returned sampler, so keep using that one.

    The mechanisms draw numbers from strata and decode them into units; the hooks below
    describe the strata, with defaults for a single stratum. A sampler that implements its
    own ``sample`` can ignore them.

    - ``_units``: how many units the budget buys; a unit is one coalition by default.
    - ``_domains``: the number of items ``N_s`` of every stratum; numbers are drawn in
      ``[0, N_s)``.
    - ``_stratum_weights``: the relative weight of every stratum; equal by default.
    - ``_complements``: the stratum holding the complements of every stratum's coalitions,
      for modifiers that pair them; unknown by default.
    - ``_enumerated``: the strata a step takes completely; none by default.
    - ``_decode``: turn the drawn numbers of their strata into units.
    - ``_stratum``: the stratum of every coalition, the inverse of ``_decode`` up to the number.
    - ``_make_sample``: build the sample from the units and their log weights.
    """

    __slots__ = ()

    def update(self, sample: SampleT, value: Float[jax.Array, "*batch"]) -> Self:
        """Return the sampler after updating it with the sample's evaluation.

        The sample's log weights correct the evaluation to an unbiased estimate of the total
        over all coalitions. The sampler may use that information to improve its design for
        the next step, e.g. by adjusting size probabilities or pairing coalitions with their
        complements.
        """
        del sample, value
        return self

    @abstractmethod
    def sample(self, budget: int) -> tuple[SampleT, Self]:
        """Draw ``budget`` coalitions per entry of the sampler's batch, with their log weights.

        The sum of ``exp(log_weights[i]) * g(S_i)`` over the coalitions estimates the sum of
        ``g`` over all coalitions: a weight is one over the coalition's expected count. A
        mechanism whose weights change with later steps says which coalitions the sum runs
        over. The sampler's batch is the sample's sharing batch: it
        broadcasts into the game batch, so a size-one axis shares its coalitions across those
        games. Index the sampler to align it, e.g. ``sampler[:, None]``.
        """

    @abstractmethod
    def log_weights(
        self, coalitions: Bool[jax.Array, "*batch n_draws n_players"]
    ) -> Float[jax.Array, "*batch n_draws"]:
        """Return the current log weight of every coalition drawn so far, over all steps.

        Summed over every coalition drawn so far, ``exp(log_weights) * g`` estimates the sum of
        ``g`` over all coalitions. The weights reflect the steps taken so far, so ask the latest
        sampler.
        """

    def _units(self, budget: int) -> int:
        """Return how many units ``budget`` coalitions buy: one coalition is one unit."""
        return budget

    @property
    def _domains(self) -> tuple[int, ...]:
        """Return the number of items ``N_s`` in every stratum."""
        raise NotImplementedError

    @property
    def _stratum_weights(self) -> Float[jax.Array, "#*batch n_strata"]:
        """Return the relative weight of every stratum; equal by default."""
        return jnp.ones(len(self._domains))

    @property
    def _complements(self) -> tuple[int, ...] | None:
        """Return the stratum holding the complements of every stratum's coalitions, if known."""
        return None

    def _enumerated(
        self,
        n_units: int,
        total: Float[jax.Array, "*batch"],
        weights: Float[jax.Array, "*batch n_strata"],
        enumerated: Bool[jax.Array, "*batch n_strata"],
    ) -> Bool[jax.Array, "*batch n_strata"]:
        """Return the strata a step of ``n_units`` takes completely; none by default.

        ``total`` counts the units after this step and ``enumerated`` the strata taken before.
        """
        del n_units, total, weights
        return jnp.zeros_like(enumerated)

    def _decode(
        self,
        numbers: UInt32[jax.Array, "*batch n_units n_words"],
        strata: Int[jax.Array, "*batch n_units"],
    ) -> Bool[jax.Array, "*batch n_units n_players"]:
        """Turn the drawn numbers of their strata into units."""
        raise NotImplementedError

    def _stratum(
        self, coalitions: Bool[jax.Array, "*batch n_draws n_players"]
    ) -> Int[jax.Array, "*batch n_draws"]:
        """Return the stratum of every coalition."""
        raise NotImplementedError

    def _make_sample(
        self,
        units: Bool[jax.Array, "*batch n_units n_players"],
        log_weights: Float[jax.Array, "*batch n_units"],
    ) -> SampleT:
        """Build the sample; by default every unit is one coalition of a ``DenseSample``."""
        # Modifiers that change the sample type override this hook.
        return cast("SampleT", DenseSample(coalitions=units, log_weights=log_weights))


# Modifiers: adjust hooks; list them first among the bases.


class Paired[OwnerT: JaxLike[Any] = "Paired[Any]"](Sampler[PairedSample, OwnerT], ABC):
    """Draw every coalition together with its complement, under either mechanism.

    A unit is the pair ``{S, ~S}``, named by one member, its anchor: the budget must be even,
    and a step draws ``budget // 2`` anchors. ``_complements`` folds the strata: a stratum
    below its complement names the pairs of both and takes both weights, a stratum that is its
    own complement names its pairs by the lower half of its numbers, and a stratum above its
    complement stays empty. The decoding must put one member of every pair into that lower
    half; bitmasks and colex do, as it holds the coalitions without the last player. A
    coalition is drawn exactly when its pair is, so both members carry the pair's weight.
    """

    __slots__ = ()

    @override
    def _units(self, budget: int) -> int:
        """Return the pairs in ``budget`` coalitions.

        Raises:
            ValueError: ``budget`` is odd and would split a pair.
        """
        if budget % 2:
            msg = f"{budget} samples would split a pair."
            raise ValueError(msg)
        return budget // 2

    def _paired_complements(self) -> tuple[np.ndarray, np.ndarray]:
        """Return every stratum and the stratum of its complements.

        Raises:
            TypeError: The design does not state its complements.
        """
        complements = self._complements
        if complements is None:
            msg = f"{type(self).__name__} does not state its complements, so it can not pair."
            raise TypeError(msg)
        return np.arange(len(complements)), np.asarray(complements)

    @property
    @override
    def _domains(self) -> tuple[int, ...]:
        """Return the number of pairs every stratum names."""
        strata, complements = self._paired_complements()
        pairs = zip(super()._domains, strata, complements, strict=True)
        return tuple(n if s < t else n // 2 if s == t else 0 for n, s, t in pairs)

    @property
    @override
    def _stratum_weights(self) -> Float[jax.Array, "#*batch n_strata"]:
        """Return the stratum weights, both strata of a pair collected on the one naming it."""
        strata, complements = self._paired_complements()
        weights = super()._stratum_weights
        both = jnp.where(strata < complements, weights + weights[..., complements], 0.0)
        return jnp.where(strata == complements, weights, both)

    @override
    def _stratum(
        self, coalitions: Bool[jax.Array, "*batch n_draws n_players"]
    ) -> Int[jax.Array, "*batch n_draws"]:
        """Return the stratum naming every coalition's pair: the lower of it and its complement."""
        _, complements = self._paired_complements()
        stratum = super()._stratum(coalitions)
        return jnp.minimum(stratum, jnp.asarray(complements)[stratum])

    @override
    def _make_sample(
        self,
        units: Bool[jax.Array, "*batch n_units n_players"],
        log_weights: Float[jax.Array, "*batch n_units"],
    ) -> PairedSample:
        """Return the anchors with their complements; every pair adds two coalitions."""
        return PairedSample(anchors=units, log_weights=jnp.repeat(log_weights, 2, axis=-1))


class Border[SampleT: Sample[Any] = DenseSample, OwnerT: JaxLike[Any] = "Border[Any, Any]"](
    Sampler[SampleT, OwnerT], ABC
):
    """Enumerate a stratum once its share of all units so far reaches ``N_s``, sample the rest.

    The units of all steps so far, less the ``N_s`` of the strata already enumerated, go to the
    other strata by their weights. A stratum whose share reaches ``N_s`` is enumerated in the
    first step with room for its ``N_s`` units and is left out afterwards. Enumerated units
    weigh one. ``WithReplacement`` asks ``_enumerated``; ``WithoutReplacement`` enumerates full
    strata by itself, so it is a ``Border`` without asking.
    """

    __slots__ = ()

    @override
    def _enumerated(
        self,
        n_units: int,
        total: Float[jax.Array, "*batch"],
        weights: Float[jax.Array, "*batch n_strata"],
        enumerated: Bool[jax.Array, "*batch n_strata"],
    ) -> Bool[jax.Array, "*batch n_strata"]:
        """Return the strata to enumerate in this step.

        Each pass adds the smallest stratum that fills up and fits the room left in this step,
        then shares the units again; one pass per stratum suffices.
        """
        sizes = _float_domains(self._domains)

        def add_one(_: int | Int[jax.Array, ""], new: jax.Array) -> jax.Array:
            taken = enumerated | new
            rest = jnp.where(taken, 0.0, weights)
            left = total[..., None] - jnp.sum(jnp.where(taken, sizes, 0.0), axis=-1, keepdims=True)
            share = left * rest / jnp.sum(rest, axis=-1, keepdims=True)  # nan once all are taken
            room = n_units - jnp.sum(jnp.where(new, sizes, 0.0), axis=-1, keepdims=True)
            fills = (rest > 0) & (share >= sizes) & (sizes <= room)
            smallest = jnp.argmin(jnp.where(fills, sizes, jnp.inf), axis=-1, keepdims=True)
            first = jnp.arange(sizes.shape[-1]) == smallest
            return new | (fills & first)

        return jax.lax.fori_loop(0, len(sizes), add_one, jnp.zeros_like(enumerated))


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class WithReplacement[
    SampleT: Sample[Any] = DenseSample,
    OwnerT: JaxLike[Any] = "WithReplacement[Any, Any]",
](Sampler[SampleT, OwnerT], ABC):
    """Draw every unit independently: a stratum, then a uniform number of it.

    A step splits its ``B`` units across strata by systematic sampling, so stratum ``s`` gets
    its expected count ``B q_s`` rounded down or up, and draws every unit's number uniformly in
    ``[0, N_s)``. Strata that ``_enumerated`` names instead take all of their ``N_s`` units,
    numbered ``0 .. N_s - 1``, before the rest is split; strata enumerated in earlier steps get
    no units. ``expected`` adds up every stratum's expected count ``E[M_s]`` over all steps, so
    a unit's log weight ``log N_s - log E[M_s]``, one over its expected count, holds for every
    unit drawn so far. Besides the shared hooks, ``sample`` runs through:

    - ``_allocate``: the counts and expected counts per stratum.
    - ``_numbers``: the number of every unit within its stratum.

    The batch is the sharing batch, one typed key (``jax.random.key``) per entry: split a key to
    run independent samplers.
    """

    keys: Key[jax.Array, "*batch"]
    # Expected units and enumerated strata over all steps; compact in the batch until the first
    # step. ``__post_init__`` widens the defaults to one entry per stratum.
    expected: Float[jax.Array, "#*batch n_strata"] = field(
        default_factory=lambda: jnp.zeros((1,)), kw_only=True
    )
    enumerated: Bool[jax.Array, "#*batch n_strata"] = field(
        default_factory=lambda: jnp.zeros((1,), bool), kw_only=True
    )
    protected_axes: ClassVar[dict[str, int]] = {"keys": 0, "expected": 1, "enumerated": 1}
    broadcasted_batch: ClassVar[set[str]] = {"expected", "enumerated"}

    def __post_init__(self) -> None:
        """Give the step state one entry per stratum, so the states of all steps stack."""
        n_strata = len(self._domains)
        for name in ("expected", "enumerated"):
            value = getattr(self, name)
            if value.shape[-1] != n_strata:
                object.__setattr__(
                    self, name, jnp.broadcast_to(value, (*value.shape[:-1], n_strata))
                )

    @property
    @abstractmethod
    @override
    def _domains(self) -> tuple[int, ...]:
        """Return the number of items ``N_s`` in every stratum."""

    @abstractmethod
    @override
    def _decode(
        self,
        numbers: UInt32[jax.Array, "*batch n_units n_words"],
        strata: Int[jax.Array, "*batch n_units"],
    ) -> Bool[jax.Array, "*batch n_units n_players"]:
        """Turn the drawn numbers of their strata into units."""

    def _allocate(
        self,
        key: Key[jax.Array, "*batch"],
        n_units: int | Float[jax.Array, "*batch"],
        weights: Float[jax.Array, "*batch n_strata"],
    ) -> tuple[Int[jax.Array, "*batch n_strata"], Float[jax.Array, "*batch n_strata"]]:
        """Split ``n_units`` across strata by ``weights``; return the counts and expected counts.

        Systematic sampling: one uniform shift places ``n_units`` markers one apart on the
        cumulated expected counts, so every count is its expected count rounded down or up.
        """
        n = jnp.asarray(n_units, dtype=weights.dtype)[..., None]
        total = jnp.sum(weights, axis=-1, keepdims=True)
        expected = jnp.where(total > 0, n * weights / total, 0.0)  # (*batch, n_strata)
        positive = weights > 0
        after = jnp.cumsum(positive[..., ::-1], axis=-1)[..., ::-1] - positive
        # The last stratum that can be drawn ends exactly at n, so rounding never lands later.
        ends = jnp.where(after == 0, n, jnp.cumsum(expected, axis=-1))
        u = _uniform(key, (1,))  # (*batch, 1); one shift per entry
        below = jnp.clip(jnp.ceil(ends - u), 0, n)  # markers below each stratum's end
        return jnp.diff(below, axis=-1, prepend=0).astype(jnp.int32), expected

    def _numbers(
        self, key: Key[jax.Array, "*batch"], strata: Int[jax.Array, "*batch n_units"]
    ) -> UInt32[jax.Array, "*batch n_units n_words"]:
        """Draw every unit's number uniformly in ``[0, N_s)`` of its stratum."""
        domains = self._domains
        width = n_words(max(domains))
        flat = strata.reshape(-1, strata.shape[-1])  # (n_keys, n_units)
        numbers = jax.vmap(lambda k, s: random_below_each(k, domains, s, width))(
            key.reshape(-1), flat
        )
        return numbers.reshape(*strata.shape, width)

    @override
    def log_weights(
        self, coalitions: Bool[jax.Array, "*batch n_draws n_players"]
    ) -> Float[jax.Array, "*batch n_draws"]:
        """Return ``log N_s - log E[M_s]`` of every coalition, with ``E[M_s]`` over all steps."""
        return _log_weights(self._domains, self._stratum(coalitions), self.expected)

    @override
    def sample(self, budget: int) -> tuple[SampleT, Self]:
        """Draw the units ``budget`` coalitions buy, each weighted by ``N_s / E[M_s]``."""
        n_units = self._units(budget)
        domains = self._domains
        shape = (*self.shape, len(domains))
        next_keys, allocate_key, number_key = _split(self.keys, 3)
        weights = jnp.broadcast_to(self._stratum_weights, shape)
        expected_before = jnp.broadcast_to(self.expected, shape)
        enumerated_before = jnp.broadcast_to(self.enumerated, shape)
        # Strata enumerated now take all of their N_s units; the rest goes to the open strata.
        total = jnp.sum(expected_before, axis=-1) + n_units
        new = self._enumerated(n_units, total, weights, enumerated_before)
        enumerated = enumerated_before | new
        sizes = _float_domains(domains)
        left = n_units - jnp.sum(jnp.where(new, sizes, 0.0), axis=-1)
        open_weights = jnp.where(enumerated, 0.0, weights)
        # Once every stratum is enumerated, further units are drawn from all of them again.
        any_open = jnp.any(open_weights > 0, axis=-1, keepdims=True)
        counts, expected = self._allocate(
            allocate_key, left, jnp.where(any_open, open_weights, weights)
        )
        capped = jnp.asarray([min(n, 2**31 - 1) for n in domains], dtype=jnp.int32)
        counts = jnp.where(new, capped, counts)
        expected = expected_before + jnp.where(new, sizes, expected)
        strata = _expand_counts(counts, n_units)  # (*batch, n_units), ascending
        # Draw the coalitions' numbers within their strata, in order in the enumerated ones
        numbers = self._numbers(number_key, strata)
        in_new = jnp.take_along_axis(new, strata, axis=-1)[..., None]
        numbers = jnp.where(in_new, _ranks(strata, numbers.shape[-1]), numbers)
        log_weights = _log_weights(domains, strata, expected)
        sample = self._make_sample(self._decode(numbers, strata), log_weights)
        return sample, replace(self, keys=next_keys, expected=expected, enumerated=enumerated)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class WithoutReplacement[
    SampleT: Sample[Any] = DenseSample,
    OwnerT: JaxLike[Any] = "WithoutReplacement[Any, Any]",
](Border[SampleT, OwnerT], ABC):
    """Draw numbers of strata ``[0, N_s)`` without replacement and decode them into units.

    Every key fixes one swap-or-not permutation per stratum, shifted by a random offset. An
    increasing internal ``cursor`` counts the units drawn so far per batch entry, compact while
    the entries agree (stacking samplers broadcasts it), and ``_allocate(cursor)`` says how many of them came from each stratum: that many positions of
    the stratum's permutation are read. A step reads the next positions of every stratum, so
    the drawn numbers are distinct and, within a stratum, uniformly distributed.

    A unit from stratum ``s`` keeps the Horvitz-Thompson log weight ``log N_s - log E[M_s]``,
    ``M_s`` of the units so far coming from ``s``. The weights recorded in a step hold for every
    unit of ``s`` up to that step, so the sum runs over all units drawn so far. Besides the
    shared hooks, ``sample`` runs through ``_allocate``: how many of the first ``total`` units
    come from each stratum, and its expectation. Positions are int32, so a sampler draws fewer
    than ``2**31`` units.

    Every stratum is capped at ``N_s``, so strata that run out are enumerated completely: the
    sampler is a ``Border`` by itself and never calls ``_enumerated``.
    """

    keys: Key[jax.Array, "*batch"]
    cursor: Int[jax.Array, "#*batch"] = field(
        default_factory=lambda: jnp.zeros((), jnp.int32), kw_only=True
    )
    protected_axes: ClassVar[dict[str, int]] = {"keys": 0, "cursor": 0}
    broadcasted_batch: ClassVar[set[str]] = {"cursor"}

    @property
    @abstractmethod
    @override
    def _domains(self) -> tuple[int, ...]:
        """Return the number of items ``N_s`` in every stratum."""

    @abstractmethod
    @override
    def _decode(
        self,
        numbers: UInt32[jax.Array, "*batch n_units n_words"],
        strata: Int[jax.Array, "*batch n_units"],
    ) -> Bool[jax.Array, "*batch n_units n_players"]:
        """Turn the drawn numbers of their strata into units."""

    def _allocate(
        self, total: Int[jax.Array, "#*batch"]
    ) -> tuple[Int[jax.Array, "*batch n_strata"], Float[jax.Array, "*batch n_strata"]]:
        """Return how many of the first ``total`` units come from each stratum, and its expectation.

        The counts sum to ``total``, stay within every stratum's ``N_s``, and never decrease
        as ``total`` grows, so a step only reads new positions. Until each of the ``K`` strata
        with positive weight holds a unit, the units visit them in an order fixed by the key:
        after ``total < K`` units, a stratum holds one with probability ``total / K``. From then
        on, Huntington-Hill on ``_stratum_weights`` fixes the counts, so the expectation is the
        count itself; strata that run out are enumerated completely.
        """
        domains = self._domains
        weights = jnp.broadcast_to(self._stratum_weights, (*self.shape, len(domains)))
        total = jnp.broadcast_to(total, self.shape)[..., None]  # (*batch, 1)
        positive = weights > 0
        n_positive = jnp.sum(positive, axis=-1, keepdims=True)  # K
        if len(domains) == 1:
            later = total
        else:
            later = jnp.broadcast_to(
                _huntington_hill(weights, domains, total[..., 0]), weights.shape
            )
        first_round = total < n_positive  # (*batch, 1)
        visited = (self._visit_order(positive) < total) & positive
        counts = jnp.where(first_round, visited, later).astype(jnp.int32)
        expected = jnp.where(first_round, positive * total / n_positive, later)
        return counts, expected.astype(float)

    def _visit_order(
        self, positive: Bool[jax.Array, "*batch n_strata"]
    ) -> Int[jax.Array, "*batch n_strata"]:
        """Return every stratum's place in the first round, a uniform order fixed by the key.

        Strata without weight come last. The key folds in ``n_strata``, apart from the keys
        that derive the permutations.
        """
        n_strata = positive.shape[-1]
        keys = jax.vmap(lambda key: jax.random.fold_in(key, n_strata))(self.keys.reshape(-1))
        u = _uniform(keys.reshape(self.shape), (n_strata,))  # (*batch, n_strata)
        return jnp.argsort(jnp.argsort(jnp.where(positive, u, jnp.inf), axis=-1), axis=-1)

    def _permutations(
        self, width: int
    ) -> tuple[
        UInt32[jax.Array, "*batch n_strata n_rounds n_words"],
        UInt32[jax.Array, "*batch n_strata n_rounds"],
        UInt32[jax.Array, "*batch n_strata n_words"],
    ]:
        """Derive every key's round keys, round seeds, and offset per stratum.

        The same key gives the same permutations. All strata draw at once, each below its own
        ``N_s``, so they permute independently; all run the rounds of the largest domain. Empty
        strata get the permutation of a single item; they are never read.
        """
        domains = self._domains
        rounds = max(n_rounds(n_items) for n_items in domains)
        strata = jnp.arange(len(domains))

        def derive(key: Key[jax.Array, ""]) -> tuple[jax.Array, jax.Array, jax.Array]:
            round_key, seed_key, offset_key = jax.random.split(key, 3)
            per_round = jnp.broadcast_to(strata[:, None], (len(domains), rounds))
            return (
                random_below_each(round_key, domains, per_round, width),  # x <-> k - x
                jax.random.bits(seed_key, (len(domains), rounds), jnp.uint32),  # coin hash seeds
                random_below_each(offset_key, domains, strata, width),  # shift after permuting
            )

        round_keys, round_seeds, offsets = jax.vmap(derive)(self.keys.reshape(-1))
        return (
            round_keys.reshape(*self.shape, *round_keys.shape[1:]),
            round_seeds.reshape(*self.shape, *round_seeds.shape[1:]),
            offsets.reshape(*self.shape, *offsets.shape[1:]),
        )

    @override
    def log_weights(
        self, coalitions: Bool[jax.Array, "*batch n_draws n_players"]
    ) -> Float[jax.Array, "*batch n_draws"]:
        """Return ``log N_s - log E[M_s]`` of every coalition, with ``E[M_s]`` at the cursor."""
        _, expected = self._allocate(self.cursor)
        return _log_weights(self._domains, self._stratum(coalitions), expected)

    @override
    def sample(self, budget: int) -> tuple[SampleT, Self]:
        """Draw the units ``budget`` coalitions buy, with their Horvitz-Thompson log weights.

        Raises:
            SamplingError: Fewer units are left than the budget buys; checked outside of jit.
        """
        n_units = self._units(budget)
        domains = self._domains
        n_items = sum(domains)
        cursor = self.cursor
        if not isinstance(cursor, Tracer) and int(jnp.max(cursor)) + n_units > n_items:
            msg = f"Only {n_items - int(jnp.max(cursor))} units are left to draw."
            raise SamplingError(msg)
        total = cursor + n_units
        before, _ = self._allocate(cursor)
        after, expected = self._allocate(total)
        counts = after - before  # this step's units per stratum
        strata = _expand_counts(counts, n_units)  # (*batch, n_units), ascending
        # Continue every stratum's permutation where the previous steps stopped.
        first = jnp.cumsum(counts, axis=-1) - counts  # each stratum's first unit in this step
        positions = jnp.take_along_axis(before - first, strata, axis=-1) + jnp.arange(n_units)
        width = n_words(max(domains))
        bounds = jnp.asarray(np.stack([from_int(n, width) for n in domains]))[strata]
        round_keys, round_seeds, offsets = self._permutations(width)
        # Draw the next positions of the permutations, then shift them by the offset and wrap around N_s.
        numbers = swap_or_not(
            from_positions(positions, width), bounds, strata, round_keys, round_seeds
        )  # (*batch, n_units, n_words)
        # Required for true uniformity.
        offset = jnp.take_along_axis(offsets, strata[..., None], axis=-2)
        numbers = add_mod(numbers, offset, bounds)
        log_weights = _log_weights(domains, strata, expected)
        sample = self._make_sample(self._decode(numbers, strata), log_weights)
        return sample, replace(self, cursor=total)


# Designs: fill the stratum hooks.


@typechecked
@dataclass(frozen=True)
class Uniform[SampleT: Sample[Any] = DenseSample, OwnerT: JaxLike[Any] = "Uniform[Any, Any]"](
    Sampler[SampleT, OwnerT], ABC
):
    """One stratum of all ``2**n_players`` coalitions, numbered by their bitmask.

    Number ``x`` is the coalition of the players whose bit is set in ``x``.
    """

    n_players: int

    @property
    @override
    def _domains(self) -> tuple[int, ...]:
        """Return ``(2**n_players,)``, the number of coalitions."""
        return (2**self.n_players,)

    @property
    @override
    def _complements(self) -> tuple[int, ...]:
        """Return ``(0,)``: the single stratum holds the complements of its coalitions."""
        return (0,)

    @override
    def _decode(
        self,
        numbers: UInt32[jax.Array, "*batch n_units n_words"],
        strata: Int[jax.Array, "*batch n_units"],
    ) -> Bool[jax.Array, "*batch n_units n_players"]:
        """Read every number's bits as its coalition."""
        del strata
        return bits(numbers, self.n_players)

    @override
    def _stratum(
        self, coalitions: Bool[jax.Array, "*batch n_draws n_players"]
    ) -> Int[jax.Array, "*batch n_draws"]:
        """Return stratum zero for every coalition."""
        return jnp.zeros(coalitions.shape[:-1], jnp.int32)


@typechecked
@dataclass(frozen=True)
class Size[SampleT: Sample[Any] = DenseSample, OwnerT: JaxLike[Any] = "Size[Any, Any]"](
    Sampler[SampleT, OwnerT], ABC
):
    """One stratum per coalition size ``s``, weighted by ``q(s)``, numbered in colex order.

    Stratum ``s`` holds the ``C(n, s)`` coalitions of size ``s``. The size weights ``q`` stay
    compact, so one distribution can serve all keys; they need not be normalized.
    """

    weights: Float[jax.Array, "#*batch n_sizes"]
    protected_axes: ClassVar[dict[str, int]] = {"weights": 1}
    broadcasted_batch: ClassVar[set[str]] = {"weights"}

    @property
    def n_players(self) -> int:
        """Return the number of players in the game."""
        return self.weights.shape[-1] - 1

    @property
    @override
    def _domains(self) -> tuple[int, ...]:
        """Return ``C(n, s)`` for every size ``s``."""
        return tuple(comb(self.n_players, size) for size in range(self.n_players + 1))

    @property
    @override
    def _stratum_weights(self) -> Float[jax.Array, "#*batch n_sizes"]:
        """Return the size weights ``q``."""
        return self.weights

    @property
    @override
    def _complements(self) -> tuple[int, ...]:
        """Return ``n - s`` for every size ``s``: complements of size-``s`` coalitions."""
        return tuple(self.n_players - size for size in range(self.n_players + 1))

    @override
    def _decode(
        self,
        numbers: UInt32[jax.Array, "*batch n_units n_words"],
        strata: Int[jax.Array, "*batch n_units"],
    ) -> Bool[jax.Array, "*batch n_units n_players"]:
        """Unrank every number as the coalition of its stratum's size, in colex order."""
        return colex_unrank(numbers, self.n_players, strata)

    @override
    def _stratum(
        self, coalitions: Bool[jax.Array, "*batch n_draws n_players"]
    ) -> Int[jax.Array, "*batch n_draws"]:
        """Return every coalition's size."""
        return jnp.sum(coalitions, axis=-1, dtype=jnp.int32)


# Helpers.


def _split(keys: Key[jax.Array, "*batch"], num: int) -> tuple[Key[jax.Array, "*batch"], ...]:
    """Split every key of the batch into ``num`` independent keys."""
    parts = jax.vmap(lambda key: jax.random.split(key, num))(keys.reshape(-1))  # (n_keys, num)
    return tuple(parts[:, i].reshape(keys.shape) for i in range(num))


def _uniform(keys: Key[jax.Array, "*batch"], shape: tuple[int, ...]) -> Float[jax.Array, "..."]:
    """Draw uniforms of ``shape`` per key, independently across the batch."""
    draws = jax.vmap(lambda key: jax.random.uniform(key, shape))(keys.reshape(-1))
    return draws.reshape(*keys.shape, *shape)


_MAX_FLOAT = 2.0**127  # every domain below fits float32


def _float_domains(domains: tuple[int, ...]) -> Float[jax.Array, " n_strata"]:
    """Return every ``N_s`` as a float; domains beyond float32 become ``inf``, never filled."""
    return jnp.asarray([float(n) if n < _MAX_FLOAT else math.inf for n in domains])


def _log_weights(
    domains: tuple[int, ...],
    strata: Int[jax.Array, "*batch n_draws"],
    expected: Float[jax.Array, "#*batch n_strata"],
) -> Float[jax.Array, "*batch n_draws"]:
    """Return ``log N_s - log E[M_s]`` of every draw's stratum: one over its expected count."""
    per_stratum = jnp.asarray([math.log(max(n, 1)) for n in domains]) - jnp.log(expected)
    shape = jnp.broadcast_shapes(per_stratum.shape[:-1], strata.shape[:-1])
    per_stratum = jnp.broadcast_to(per_stratum, (*shape, len(domains)))
    strata = jnp.broadcast_to(strata, (*shape, strata.shape[-1]))
    return jnp.take_along_axis(per_stratum, strata, axis=-1)


def _expand_counts(
    counts: Int[jax.Array, "*batch n_strata"], budget: int
) -> Int[jax.Array, "*batch budget"]:
    """Expand the counts per stratum into the stratum of each draw, in ascending order.

    ``budget`` must be the total of every count row, fixing the output shape under jit.
    """
    ends = jnp.cumsum(counts, axis=-1)  # (*batch, n_strata); where each stratum's draws end
    # Draw i lies in stratum s once it has passed the ends of the first s strata.
    return jnp.sum(jnp.arange(budget)[:, None] >= ends[..., None, :], axis=-1, dtype=jnp.int32)


def _ranks(
    strata: Int[jax.Array, "*batch n_units"], width: int
) -> UInt32[jax.Array, "*batch n_units width"]:
    """Return every unit's rank within its stratum, as words; strata must be ascending."""
    units = jnp.arange(strata.shape[-1])
    starts = jnp.concatenate(
        [jnp.ones_like(strata[..., :1], bool), strata[..., 1:] != strata[..., :-1]], axis=-1
    )  # the first unit of every stratum
    first = jax.lax.cummax(jnp.where(starts, units, 0), axis=strata.ndim - 1)
    return from_positions(units - first, width)


_MAX_COUNT = 2**24  # float32 represents every count up to here exactly


def _huntington_hill(
    weights: Float[jax.Array, "#*batch n_strata"],
    capacities: tuple[int, ...],
    total: Int[jax.Array, "#*batch"],
) -> Int[jax.Array, "#*batch n_strata"]:
    """Split ``total`` draws across strata by Huntington-Hill, within every stratum's capacity.

    Draws go one at a time to the open stratum with the highest priority
    ``q_s / sqrt(M_s (M_s + 1))``, ties to the lowest stratum: every stratum with ``q_s > 0``
    gets a draw first, and more draws never take one away. Computed directly: scale ``q`` by
    ``lam``, round ``q_s * lam`` up above the geometric mean of its neighbours, and search the
    largest ``lam`` whose counts fit ``total``.
    If the exact ``total`` can not be reached, the left over draws go to the strata which move up at the jump, lower ones first.
    For numerical stability, caps are capped at ``_MAX_COUNT`` and the weights are normalized to sum to one.
    """
    q = weights / jnp.sum(weights, axis=-1, keepdims=True)
    caps = jnp.asarray([min(c, _MAX_COUNT) for c in capacities], dtype=q.dtype)
    target = jnp.asarray(total, dtype=q.dtype)[..., None]  # (#*batch, 1)

    def counts(log_lam: jax.Array) -> jax.Array:
        x = q * jnp.exp(log_lam)[..., None]
        k = jnp.floor(x)
        above = x * x > k * (k + 1)  # x lies above the geometric mean of k and k + 1
        return jnp.minimum(k + above, caps)

    def fit(c: jax.Array) -> jax.Array:
        return jnp.sum(c, axis=-1, keepdims=True) <= target

    # At lo every positive stratum holds one draw; at hi every stratum reaches its capacity.
    lo = jnp.log(0.5 * jnp.sqrt(2.0) / jnp.max(q, axis=-1))
    hi = jnp.log((target[..., 0] + 1) / jnp.min(jnp.where(q > 0, q, jnp.inf), axis=-1)) + 1
    first, last = counts(lo), counts(hi)
    below = jnp.where(fit(first), first, 0.0)  # fewer draws than positive strata: none given yet

    def bisect(_: int | Int[jax.Array, ""], state: tuple) -> tuple:
        lo, hi, below, above = state
        mid = (lo + hi) / 2
        c = counts(mid)
        fits = fit(c)
        return (
            jnp.where(fits[..., 0], mid, lo),
            jnp.where(fits[..., 0], hi, mid),
            jnp.where(fits, c, below),
            jnp.where(fits, above, c),
        )

    # Keep the counts of both bounds from inside the loop: recomputing them at a bound that sits
    # on a tie can round the other way.
    _, _, below, above = jax.lax.fori_loop(0, 64, bisect, (lo, hi, below, last))
    gaining = above > below  # the strata whose next draw comes at the jump past ``lo``
    left = target - jnp.sum(below, axis=-1, keepdims=True)
    return (below + (gaining & (jnp.cumsum(gaining, axis=-1) <= left))).astype(jnp.int32)


# Samplers: modifiers, design, mechanism.


@typechecked
@dataclass(frozen=True, slots=True)
class SizeWithReplacement(
    Size[DenseSample, "SizeWithReplacement"], WithReplacement[DenseSample, "SizeWithReplacement"]
):
    """Draw a size with probability ``q(s)``, then a coalition of that size uniformly."""


@typechecked
@dataclass(frozen=True, slots=True)
class SizeWithoutReplacement(
    Size[DenseSample, "SizeWithoutReplacement"],
    WithoutReplacement[DenseSample, "SizeWithoutReplacement"],
):
    """Draw coalitions without replacement, split across sizes by Huntington-Hill on ``q``.

    Every size with ``q(s) > 0`` gets a draw first, sizes that run out are enumerated
    completely, and later steps never take a draw back.
    """


@typechecked
@dataclass(frozen=True, slots=True)
class UniformWithReplacement(
    Uniform[DenseSample, "UniformWithReplacement"],
    WithReplacement[DenseSample, "UniformWithReplacement"],
):
    """Draw coalitions uniformly with replacement; every draw keeps ``n log 2``."""


@typechecked
@dataclass(frozen=True, slots=True)
class UniformWithoutReplacement(
    Uniform[DenseSample, "UniformWithoutReplacement"],
    WithoutReplacement[DenseSample, "UniformWithoutReplacement"],
):
    """Draw coalitions uniformly without replacement; every draw keeps ``n log 2``."""


@typechecked
@dataclass(frozen=True, slots=True)
class BorderSizeWithReplacement(
    Border[DenseSample, "BorderSizeWithReplacement"],
    Size[DenseSample, "BorderSizeWithReplacement"],
    WithReplacement[DenseSample, "BorderSizeWithReplacement"],
):
    """Enumerate the sizes a step would draw at least ``C(n, s)`` times, sample the others."""


@typechecked
@dataclass(frozen=True, slots=True)
class PairedSizeWithReplacement(
    Paired["PairedSizeWithReplacement"],
    Size[PairedSample, "PairedSizeWithReplacement"],
    WithReplacement[PairedSample, "PairedSizeWithReplacement"],
):
    """Draw pairs ``{S, ~S}`` with replacement, by the pair weights ``q(s) + q(n - s)``."""


@typechecked
@dataclass(frozen=True, slots=True)
class PairedBorderSizeWithReplacement(
    Paired["PairedBorderSizeWithReplacement"],
    Border[PairedSample, "PairedBorderSizeWithReplacement"],
    Size[PairedSample, "PairedBorderSizeWithReplacement"],
    WithReplacement[PairedSample, "PairedBorderSizeWithReplacement"],
):
    """Draw pairs with replacement, enumerating the pair sizes that fill up."""


@typechecked
@dataclass(frozen=True, slots=True)
class PairedSizeWithoutReplacement(
    Paired["PairedSizeWithoutReplacement"],
    Size[PairedSample, "PairedSizeWithoutReplacement"],
    WithoutReplacement[PairedSample, "PairedSizeWithoutReplacement"],
):
    """Draw pairs without replacement, stratified by the smaller size of the pair.

    The pair strata are the sizes up to ``n / 2``, with weights ``q(s) + q(n - s)``; for even
    ``n``, size ``n / 2`` names its ``C(n, n / 2) / 2`` pairs by the coalitions without the last
    player.
    """


@typechecked
@dataclass(frozen=True, slots=True)
class PairedUniformWithoutReplacement(
    Paired["PairedUniformWithoutReplacement"],
    Uniform[PairedSample, "PairedUniformWithoutReplacement"],
    WithoutReplacement[PairedSample, "PairedUniformWithoutReplacement"],
):
    """Draw pairs uniformly without replacement, named by the bitmasks below ``2**(n - 1)``."""
