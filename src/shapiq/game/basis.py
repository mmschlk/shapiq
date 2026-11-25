"""K-additive games stored as coefficients on interaction sets of a game basis."""

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, override

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Bool, Float, Shaped
from ndlike.jax_functions import jax_multiply, jax_sum, jax_zeros_like
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.game.finite import BoolFinitePlayerGame, _FiniteGame
from shapiq.game.kadditive import KAdditiveGame
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class BasisGame[
    P,
    V: JaxLike[Any],
    OwnerT: "BasisGame[Any, Any, Any]" = "BasisGame[P, V, Any]",
](
    BoolFinitePlayerGame[P, JaxValue],
    KAdditiveGame[P, JaxValue],
    JaxAxisProtected[OwnerT, Any, _FiniteGame[P]],
):
    """A batch of k-additive games stored as basis coefficients on interaction sets.

    ``values[..., t]`` is the coefficient of the basis function of ``interactions[..., t, :]``, and
    v(S) = sum over T of values[T] * basis_T(S). Sets not in ``interactions`` have coefficient
    zero. Subclasses fix the basis through ``_basis`` and decide whether ``interactions`` and
    ``additivity`` are stored fields or implicit properties.

    ``interactions`` may be shared across the batch or batched along axes that broadcast into
    the game batch without enlarging it. ``additivity`` is static metadata, checked against
    ``interactions`` when they are concrete.
    """

    values: Float[JaxValue, "*batch n_interactions"]
    # A required slot replaces the inherited abstract players property.
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players"}

    @property
    @abstractmethod
    def interactions(self) -> Bool[jax.Array, "#*batch n_interactions n_players"]:
        """Return the interaction sets whose coefficients ``values`` holds."""

    def __post_init__(self) -> None:
        """Check that no interaction exceeds the declared additivity, when values are concrete.

        Raises:
            TypeError: ``additivity`` is a boolean rather than an integer.
            ValueError: The bound is negative or an interaction is larger than ``additivity``.
        """
        if isinstance(self.additivity, bool):
            msg = "Additivity must be an integer, not bool."
            raise TypeError(msg)
        if self.additivity < 0:
            msg = "Additivity must be nonnegative."
            raise ValueError(msg)
        sizes = jnp.sum(self.interactions, axis=-1)
        # Inside jit the interactions are traced and cannot be checked.
        if not isinstance(sizes, Tracer) and sizes.size and int(jnp.max(sizes)) > self.additivity:
            msg = "An interaction is larger than the declared additivity."
            raise ValueError(msg)

    @override
    def _prepare_assignment_value(self, value: object, *, method: JaxAtMethod) -> object:
        """Require compatible basis type and bounded additivity on indexed replacement."""
        del method
        if isinstance(value, BasisGame):
            if not isinstance(value, type(self)):
                msg = f"Cannot assign {type(value).__name__} into {type(self).__name__}."
                raise TypeError(msg)
            if value.additivity > self.additivity:
                msg = (
                    f"Cannot assign {type(value).__name__} with additivity {value.additivity} "
                    f"into {type(self).__name__} with additivity {self.additivity}."
                )
                raise ValueError(msg)
        return value

    @classmethod
    @override
    def _combine_sequence_metadata(
        cls,
        items: Sequence["BasisGame[Any, Any, Any]"],
        *,
        func: Callable[..., Any] | None,
    ) -> dict[str, Any]:
        """Combine declared additivity bounds order-independently via maximum."""
        del func
        return {"additivity": max(item.additivity for item in items)}

    @abstractmethod
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_interactions"]:
        """Return basis_T(S) for every listed interaction T and coalition S."""

    @typechecked
    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Sum the coefficients weighted by their basis functions at each coalition."""
        return jax_sum(jax_multiply(self.values, self._basis(coalition)), axis=-1)

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[JaxValue, "..."]:
        """Return zeros of the coefficients' type without evaluating coalitions."""
        if isinstance(self.values, jax.Array):
            return cast("Shaped[JaxValue, '...']", jnp.zeros(shape, dtype=self.dtype))
        return cast("Shaped[JaxValue, '...']", jax_zeros_like(self.values, shape=shape))
