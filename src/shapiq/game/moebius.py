"""K-additive games stored as Möbius coefficients."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, override

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Bool, Shaped
from ndlike.jax_like import JaxLike

from shapiq.game import Additive
from shapiq.game.basis import BasisGame
from shapiq.typing import typechecked


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class MoebiusGame[P, V: JaxLike](BasisGame[P, V]):
    """A batch of k-additive games stored as Möbius coefficients.

    ``values[..., t]`` is the Möbius coefficient m(T_t) of ``interactions[t]``, and
    v(S) = sum over T ⊆ S of m(T). Concrete subclasses decide how the sets are stored.
    """

    @override
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.players.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_interactions"]:
        """Return whether every interaction T is contained in S (T ⊆ S)."""
        return ~jnp.any(self.interactions & ~coalition[..., None, :], axis=-1)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SetMoebiusGame[P, V: JaxLike](MoebiusGame[P, V]):
    """A batch of k-additive games with Möbius coefficients on explicitly stored sets.

    Sets not in ``interactions`` have coefficient zero.
    """

    interactions: Bool[jax.Array, "n_interactions {players.n_players}"] = field()
    additivity: int = field()
    protected_axes: ClassVar[dict[str, int]] = {"interactions": 2}
    broadcasted_batch: ClassVar[set[str]] = {"interactions"}


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class AdditiveMoebiusGame[P, V: JaxLike](SetMoebiusGame[P, V], Additive):
    """A batch of additive games: ``values[..., t]`` is the Möbius coefficient of ``interactions[t]``.

    The interactions are the singletons in any order, each player exactly once, so
    ``values[..., t]`` belongs to the player in ``interactions[t]``, not to position ``t``.
    """

    additivity: Literal[1] = field(default=1, init=False)

    def __post_init__(self) -> None:
        """Check that the interactions are a permutation of the singletons, when concrete.

        Raises:
            ValueError: An interaction is no singleton, or a player is missing or repeated.
        """
        SetMoebiusGame.__post_init__(self)
        if not isinstance(self.interactions, Tracer):
            rows_single = jnp.all(jnp.sum(self.interactions, axis=-1) == 1)
            each_player_once = jnp.all(jnp.sum(self.interactions, axis=-2) == 1)
            if not (rows_single and each_player_once):
                msg = "AdditiveMoebiusGame interactions must be a permutation of the singletons."
                raise ValueError(msg)

    @override
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.players.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_interactions"]:
        """Matches the player of each singleton interaction to the coalition, ignoring order."""
        return jnp.take(coalition, jnp.argmax(self.interactions, axis=-1), axis=-1)
