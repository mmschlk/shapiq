from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp

from shapiq.sampling._schedule import UnitScheduleSampler

if TYPE_CHECKING:
    from jax import Array


@runtime_checkable
class AntitheticDraws(Protocol):
    """Optional hook declaring what pairing means for a sampler.

    Samplers that render structured units (permutation walks) implement the
    draw triad so ``PairedSampler`` can render the antithesis as a bona-fide
    unit; samplers without the hook are paired by complementing their
    rendered rows.
    """

    def unit_draw(self, unit_index: int) -> Array:
        """Return the raw draw of one sampled unit."""
        ...

    def render_draw(self, draw: Array) -> Array:
        """Return the dense coalition masks a draw stands for."""
        ...

    def antithetic_draw(self, draw: Array) -> Array:
        """Return the variance-reducing antithesis of a draw."""
        ...


class PairedSampler(UnitScheduleSampler):
    """Pairing as sampler composition: ``PairedSampler(sampler)``.

    The wrapper owns the emission schedule (seed block, budgets, pending
    units) and uses the wrapped sampler purely as a unit renderer, so any
    ``UnitScheduleSampler`` gains variance-reducing pairing without knowing
    about it. Every unit contains the wrapped sampler's unit followed by its
    antithesis: samplers implementing the ``AntitheticDraws`` hook define
    what the antithesis is (the reversed permutation for walks); all others
    are paired by complementing the rendered rows, which is exact for
    single-coalition units. The sampling quantum doubles.
    """

    def __init__(self, sampler: UnitScheduleSampler) -> None:
        """Wrap a sampler so every unit also renders its antithesis.

        Args:
            sampler: The sampler whose units are paired. Must be a
                ``UnitScheduleSampler``; pairing composes on the schedule.

        Raises:
            TypeError: If the sampler is already paired or is no
                unit-schedule sampler.
        """
        if isinstance(sampler, PairedSampler):
            msg = "the sampler is already paired: pairing twice would pair antitheses"
            raise TypeError(msg)
        if not isinstance(sampler, UnitScheduleSampler):
            msg = (
                "PairedSampler composes on unit-schedule samplers, got "
                f"{type(sampler).__name__}"
            )
            raise TypeError(msg)
        super().__init__(
            sampler.n_players,
            sampler.target_shape,
            share_samples=sampler.share_samples,
            random_state=0,
        )
        self.sampler = sampler

    @property
    def sampling_quantum(self) -> int:
        """Return the unit length: the wrapped quantum, doubled."""
        return 2 * self.sampler.sampling_quantum

    @property
    def n_seed_samples(self) -> int:
        """Return the wrapped sampler's seed block length; seeds are not paired."""
        return self.sampler.n_seed_samples

    def __getattr__(self, name: str) -> object:
        """Expose the wrapped sampler's metadata (order, walk_length, ...)."""
        if name.startswith("_"):
            msg = f"{type(self).__name__!r} object has no attribute {name!r}"
            raise AttributeError(msg)
        return getattr(self.sampler, name)

    def _seed_masks(self) -> Array:
        """Return the wrapped sampler's deterministic seed block."""
        return self.sampler._seed_masks()  # noqa: SLF001 - rendering the wrapped units

    def _sampled_unit_masks(self, unit_index: int) -> Array:
        """Render the wrapped unit followed by its antithesis."""
        if isinstance(self.sampler, AntitheticDraws):
            draw = self.sampler.unit_draw(unit_index)
            rendered = self.sampler.render_draw(draw)
            antithetic = self.sampler.render_draw(self.sampler.antithetic_draw(draw))
        else:
            # base case of classical pairing of sampling complement coalitions
            rendered = self.sampler._sampled_unit_masks(unit_index)  # noqa: SLF001 - wrapped unit
            antithetic = ~rendered
        return jnp.concatenate([rendered, antithetic], axis=-2)
