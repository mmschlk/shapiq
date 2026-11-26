"""K-additive games stored as coefficients on interaction sets of a game basis."""

import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

import jax
import jax.numpy as jnp
from flextype import CastUnavailable
from jax.core import Tracer
from jaxtyping import Bool, Float, Shaped
from ndlike.jax_functions import jax_multiply, jax_sum, jax_zeros_like
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.game.finite import BoolFinitePlayerGame, _FiniteGame
from shapiq.game.kadditive import KAdditiveGame
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value._dtype import cast_value, require_value_spec
from shapiq.value.value import JaxValue

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike


def _require_table_values(values: object) -> jax.Array:
    """Validate that stored basis coefficients are a floating ``jax.Array`` for ``TableGame``."""
    if not isinstance(values, jax.Array) or not jnp.issubdtype(values.dtype, jnp.floating):
        msg = (
            f"TableGame requires floating jax.Array values, got {type(values).__name__} "
            f"(dtype={getattr(values, 'dtype', None)})."
        )
        raise CastUnavailable(msg)
    return values


def _check_table_n_players(n_players: int) -> jnp.dtype:
    """Validate that ``2**n_players`` fits in JAX's enabled signed index/dimension width."""
    index_dtype = jnp.dtype(jnp.int64 if jax.config.x64_enabled else jnp.int32)
    # Both coalition indices (up to 2**n - 1) and axis length 2**n must fit in signed index_dtype.
    max_players = jnp.iinfo(index_dtype).bits - 2
    if n_players > max_players:
        msg = "Table indices exceed the enabled integer width."
        raise ValueError(msg)
    return index_dtype


def _scatter_basis_coefficients(
    values: Float[jax.Array, "*batch n_interactions"],
    interactions: Bool[jax.Array, "#*batch n_interactions {n_players}"],
    n_players: int,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Scatter basis coefficients into a dense ``(*batch, 2**n_players)`` accumulation table.

    Coalition indices follow ``TableGame``'s little-endian bit order where bit ``j``
    corresponds to local player ``j``. Duplicate interactions are summed via additive
    scatter in the stored value dtype.
    """
    index_dtype = _check_table_n_players(n_players)
    batch_shape = values.shape[:-1]
    n_interactions = values.shape[-1]
    if n_players == 0:
        return jnp.sum(values, axis=-1, keepdims=True, dtype=values.dtype)
    n_coalitions = 1 << n_players
    if n_interactions == 0 or 0 in batch_shape:
        return jnp.zeros((*batch_shape, n_coalitions), dtype=values.dtype)

    weights = jnp.left_shift(
        jnp.asarray(1, dtype=index_dtype),
        jnp.arange(n_players, dtype=index_dtype),
    )
    indices = jnp.sum(interactions.astype(index_dtype) * weights, axis=-1, dtype=index_dtype)

    if math.prod(interactions.shape[:-2]) == 1:
        table = jnp.zeros((*batch_shape, n_coalitions), dtype=values.dtype)
        return table.at[..., indices.reshape(n_interactions)].add(values)

    batch_size = math.prod(batch_shape)
    batch_idx = jnp.arange(batch_size)[:, None]
    indices_flat = jnp.broadcast_to(indices, (*batch_shape, n_interactions)).reshape(
        batch_size, n_interactions
    )
    values_flat = values.reshape(batch_size, n_interactions)
    table_flat = jnp.zeros((batch_size, n_coalitions), dtype=values.dtype)
    table_flat = table_flat.at[batch_idx, indices_flat].add(values_flat)
    return table_flat.reshape(*batch_shape, n_coalitions)


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

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return the coefficient dtype, strong after summing the basis terms."""
        return require_value_spec(self.values).update(weak_type=False)

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the stored coefficient dtype."""
        return self.values.dtype

    @override
    def astype(
        self,
        dtype: "DTypeLike | NpDTypeLike | None",
        copy: bool = False,
        device: "jax.Device | Sharding | None" = None,
    ) -> Self:
        """Cast the stored coefficients."""
        game = self if device is None else self.to_device(device)
        target = None if dtype is None else jax.dtypes.canonicalize_dtype(jnp.dtype(dtype))
        values = cast_value(game.values, target, copy=copy, device=None)
        return game if values is game.values else replace(game, values=values)

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
