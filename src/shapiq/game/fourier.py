"""K-additive games stored as Fourier coefficients on listed interaction sets."""

from dataclasses import dataclass, field
from typing import ClassVar, override

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Shaped
from ndlike.jax_like import JaxLike

from shapiq.game.basis import BasisGame
from shapiq.typing import typechecked


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class FourierGame[P, V: JaxLike](BasisGame[P, V]):
    """A batch of k-additive games with Fourier coefficients on explicitly stored sets.

    ``values[..., t]`` is the Fourier coefficient f(T_t) of ``interactions[t]``, and
    v(S) = sum over stored T of f(T) * (-1)**|T & S|. Unlisted sets have coefficient zero;
    the inverse transform is f(T) = 2**-n * sum over all S of v(S) * (-1)**|T & S|.

    Fourier and Möbius degrees coincide, so ``additivity`` bounds both.
    """

    interactions: Bool[jax.Array, "n_interactions {players.n_players}"] = field()
    additivity: int = field()
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "interactions": 2, "players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players", "interactions"}

    @override
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.players.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_interactions"]:
        """Return the parity (-1)**|T & S| of every listed interaction T."""
        return (-1) ** jnp.sum(self.interactions & coalition[..., None, :], axis=-1)
