from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp

from shapiq.coalitions import DenseCoalitionArray
from shapiq.sampling._base import LawfulSampler, Sampler

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from shapiq.coalitions import CoalitionArray


@runtime_checkable
class AntitheticDraws(Protocol):
    """Optional capability declaring what the antithesis of a draw is.

    The meaning is draw-kind knowledge and lives on the sampler: reversal
    for permutations, complement for coalitions. ``PairedSampler`` attaches
    to any sampler declaring it.
    """

    def antithetic(self, draws: Array) -> Array:
        """Return the variance-reducing antithesis of each draw."""
        ...


class PairedSampler(Sampler):
    """Pairing as sampler composition: ``PairedSampler(sampler)``.

    Attachable to any sampler that declares its antithesis: every unit
    index yields the wrapped draw followed by its antithesis, so one unit
    carries both. The wrapped sampler defines what the antithesis means —
    the reversed permutation for permutation draws, the complement for
    coalition draws — and the wrapper stays a thin draw transformer.
    """

    def __init__(self, sampler: Sampler) -> None:
        """Wrap a sampler so every unit also draws its antithesis.

        Args:
            sampler: The sampler whose draws are paired. Must declare an
                antithesis.

        Raises:
            TypeError: If the sampler is already paired, or declares no
                antithesis.
        """
        if isinstance(sampler, PairedSampler):
            msg = "the sampler is already paired: pairing twice would pair antitheses"
            raise TypeError(msg)
        if not isinstance(sampler, AntitheticDraws):
            msg = (
                f"{type(sampler).__name__} declares no antithesis; pairing needs the "
                "sampler to define what the antithetic draw is"
            )
            raise TypeError(msg)
        super().__init__(
            sampler.n_players,
            sampler.target_shape,
            share_samples=sampler.share_samples,
            random_state=0,
        )
        self.sampler = sampler
        # the marginal law of one paired draw is the antithesis-symmetrized
        # wrapped law; grafted as an instance attribute exactly when the
        # wrapped sampler declares a law, so the LawfulSampler capability
        # check sees through pairing (python 3.12+ protocol isinstance uses
        # getattr_static, which ignores __getattr__)
        if isinstance(sampler, LawfulSampler):
            self.log_probability = partial(
                _paired_log_probability,
                sampler.log_probability,
                sampler.antithetic,
            )

    @property
    def draws_per_unit(self) -> int:
        """Return the wrapped count, doubled: draw and antithesis per unit."""
        return 2 * self.sampler.draws_per_unit

    def draws(self, unit_indices: Array) -> Array:
        """Return draw and antithesis per unit, interleaved along the unit axis."""
        drawn = self.sampler.draws(unit_indices)
        antithetic = self.sampler.antithetic(drawn)  # type: ignore[attr-defined]
        paired = jnp.stack([drawn, antithetic], axis=1)
        return paired.reshape(-1, *drawn.shape[1:])

    def __getattr__(self, name: str) -> object:
        """Expose the wrapped sampler's metadata (p, size probabilities, ...).

        The law is never forwarded verbatim: when the wrapped sampler
        declares one, the symmetrized law is grafted at construction and
        found before this fallback; otherwise the paired law is undeclared.
        """
        if name == "log_probability" or name.startswith("_"):
            msg = f"{type(self).__name__!r} object has no attribute {name!r}"
            raise AttributeError(msg)
        return getattr(self.sampler, name)


def _paired_log_probability(
    inner_law: Callable[[CoalitionArray], Array],
    inner_antithetic: Callable[[Array], Array],
    coalitions: CoalitionArray,
) -> Array:
    """Return the antithesis-symmetrized law of one paired draw."""
    dense = jnp.asarray(coalitions.to_dense())
    antitheses = DenseCoalitionArray(inner_antithetic(dense))
    return jnp.logaddexp(inner_law(coalitions), inner_law(antitheses)) - jnp.log(2.0)
