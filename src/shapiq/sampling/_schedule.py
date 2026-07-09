from __future__ import annotations

from abc import abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
from jax import Array

from shapiq.coalitions import DenseCoalitionArray
from shapiq.sampling._base import Sampler

if TYPE_CHECKING:
    from shapiq._shape import ShapeLike
    from shapiq.coalitions import CoalitionArray
    from shapiq.sampling._base import ShareSamples
    from shapiq.sampling._state import ApproximationState


class UnitScheduleSampler(Sampler["ApproximationState"]):
    """Base sampler emitting a seed block followed by fixed-size sampled units.

    The emission schedule is a one-time deterministic seed block (by default
    the empty and grand coalition) followed by sampled units of
    ``sampling_quantum`` coalitions each; constructing a sampler or an
    approximator therefore never evaluates a game. Budgets are spent exactly:
    a unit cut short by the budget stays pending and is resumed by the
    evolved sampler. Units derive their randomness from
    ``fold_in(random_state, unit_index)``, so sampling does not depend on how
    a budget is split across calls.
    """

    _units_started: int
    _pending_pos: int

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        *,
        share_samples: ShareSamples = False,
        random_state: Array | int = 0,
    ) -> None:
        """Initialize a unit-schedule sampler.

        Args:
            n_players: Number of players in the explained game. Must be at
                least two.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            random_state: Integer seed or JAX PRNG key used to derive the
                sampled units.

        Raises:
            ValueError: If ``n_players`` is smaller than two.
            TypeError: If ``random_state`` is neither an integer nor a JAX
                PRNG key.
        """
        super().__init__(n_players, target_shape, share_samples=share_samples)
        if self.n_players < 2:
            msg = "sampled units require at least two players"
            raise ValueError(msg)
        self._key = _validate_random_state(random_state)
        self._units_started = 0
        self._pending_pos = 0

    @property
    def n_seed_samples(self) -> int:
        """Return the length of the deterministic seed block."""
        return 2

    @property
    def n_pending_samples(self) -> int:
        """Return the number of emitted coalitions of the unfinished unit.

        The unfinished unit is either the seed block or the current sampled
        unit; its already-emitted coalitions stay pending until a later
        sample call completes the unit.
        """
        return self._pending_pos

    def _sample(
        self,
        state: ApproximationState,  # noqa: ARG002 - schedule samplers are not adaptive
        budget: int,
    ) -> tuple[CoalitionArray, Self]:
        """Emit exactly budget coalitions, resuming any pending unit."""
        chunks: list[Array] = []
        units = self._units_started
        position = self._pending_pos
        remaining = budget
        if position > 0:
            masks = self._unit_masks(units - 1)
            length = masks.shape[-2]
            take = min(length - position, remaining)
            chunks.append(masks[..., position : position + take, :])
            position = (position + take) % length
            remaining -= take
        if remaining > 0 and units == 0:
            masks = self._unit_masks(0)
            length = masks.shape[-2]
            take = min(length, remaining)
            chunks.append(masks[..., :take, :])
            units = 1
            position = take % length
            remaining -= take
        quantum = self.sampling_quantum
        full_units = remaining // quantum
        if full_units > 0:
            batch = self._sampled_unit_batch(jnp.arange(units - 1, units - 1 + full_units))
            if batch.shape[-2] != quantum:
                msg = (
                    f"sampled units hold {batch.shape[-2]} coalitions but sampling_quantum "
                    f"is {quantum}; unit-schedule samplers emit quantum-sized units"
                )
                raise ValueError(msg)
            chunks.append(_flatten_units(batch))
            units += full_units
            remaining -= full_units * quantum
        if remaining > 0:
            masks = self._unit_masks(units)
            chunks.append(masks[..., :remaining, :])
            units += 1
            position = remaining
        coalitions = DenseCoalitionArray(jnp.concatenate(chunks, axis=-2))
        return coalitions, self._evolve(units_started=units, pending_pos=position)

    def _evolve(self, *, units_started: int, pending_pos: int) -> Self:
        """Return a sampler that resumes after the emitted coalitions."""
        evolved = copy(self)
        evolved._units_started = units_started  # noqa: SLF001 - evolving a copy of self
        evolved._pending_pos = pending_pos  # noqa: SLF001 - evolving a copy of self
        return evolved

    def _unit_masks(self, unit_index: int) -> Array:
        """Return the dense coalition masks of one schedule unit."""
        if unit_index == 0:
            return jnp.broadcast_to(
                self._seed_masks(),
                (*self.shared_target_shape, self.n_seed_samples, self.n_players),
            )
        return self._sampled_unit_masks(unit_index - 1)

    def _seed_masks(self) -> Array:
        """Return the deterministic seed block masks."""
        return jnp.stack(
            [jnp.zeros(self.n_players, dtype=bool), jnp.ones(self.n_players, dtype=bool)],
        )

    @abstractmethod
    def _sampled_unit_masks(self, unit_index: int) -> Array:
        """Return the dense coalition masks of one full sampled unit."""

    def _sampled_unit_batch(self, unit_indices: Array) -> Array:
        """Return the masks of many sampled units, stacked on a new leading axis.

        The default renders units one by one, so custom samplers stay correct
        without extra work; shipped samplers override it to generate the whole
        batch in a few vectorized dispatches. Overrides must be bit-identical
        to the sequential per-unit stream: budgets may be split arbitrarily,
        and every unit must render the same masks whether it is generated
        alone or inside a batch.
        """
        return jnp.stack([self._sampled_unit_masks(index) for index in unit_indices.tolist()])

    def _unit_keys(self, unit_indices: Array) -> Array:
        """Return the PRNG keys of many sampled units in one dispatch.

        Folding the sampler key by unit index is what makes unit generation
        order-free; the batched fold is bit-identical to folding per unit.
        """
        return jax.vmap(lambda index: jax.random.fold_in(self._key, index))(unit_indices)


def _flatten_units(batch: Array) -> Array:
    """Merge a leading unit axis into the sample axis, preserving unit order."""
    stacked = jnp.moveaxis(batch, 0, -3)
    return stacked.reshape(*stacked.shape[:-3], -1, stacked.shape[-1])


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
