"""K-additive games stored as coefficients on interaction sets of a game basis."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, override

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Bool, Float, Shaped
from ndlike.jax_functions import jax_multiply, jax_sum
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game import BoolFinitePlayerGame, KAdditive
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class BasisGame[P, V: JaxLike](
    BoolFinitePlayerGame[P, JaxValue], JaxAxisProtected[jax.Array], KAdditive
):
    """A batch of k-additive games stored as basis coefficients on interaction sets.

    ``values[..., t]`` is the coefficient of the basis function of ``interactions[t]``, and
    v(S) = sum over T of values[T] * basis_T(S). Sets not in ``interactions`` have coefficient
    zero. Subclasses fix the basis through ``_basis`` and decide whether ``interactions`` and
    ``additivity`` are stored fields or implicit properties.

    ``additivity`` is static metadata, checked against ``interactions`` when they are concrete.

    Subclasses must re-declare every field: ndlike 0.3.0 checks ``protected_axes`` against the
    subclass's own annotations only. A re-declared slot also replaces an abstract property.
    """

    values: Float[jax.Array, "*batch n_interactions"]
    # A required slot replaces the inherited abstract players property.
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players"}

    @property
    @abstractmethod
    def interactions(self) -> Bool[jax.Array, "n_interactions n_players"]:
        """Return the interaction sets whose coefficients ``values`` holds."""

    @property
    @abstractmethod
    def additivity(self) -> int:
        """Return a nonnegative upper bound on the interaction order."""

    def __post_init__(self) -> None:
        """Check that no interaction exceeds the declared additivity, when values are concrete.

        Raises:
            ValueError: The bound is negative or an interaction is larger than ``additivity``.
        """
        if self.additivity < 0:
            msg = "Additivity must be nonnegative."
            raise ValueError(msg)
        sizes = jnp.sum(self.interactions, axis=-1)
        # Inside jit the interactions are traced and cannot be checked.
        if not isinstance(sizes, Tracer) and sizes.size and int(jnp.max(sizes)) > self.additivity:
            msg = "An interaction is larger than the declared additivity."
            raise ValueError(msg)

    @abstractmethod
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.players.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_interactions"]:
        """Return basis_T(S) for every listed interaction T and coalition S."""

    @typechecked
    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.players.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Sum the coefficients weighted by their basis functions at each coalition."""
        return jax_sum(jax_multiply(self.values, self._basis(coalition)), axis=-1)
