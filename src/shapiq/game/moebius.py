"""K-additive games stored as Möbius coefficients."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, override

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Shaped
from ndlike.jax_like import JaxLike

from shapiq.game import Additive
from shapiq.game.basis import BasisGame
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class MoebiusGame[P, V: JaxLike](BasisGame[P, V]):
    """A batch of k-additive games stored as Möbius coefficients.

    ``values[..., t]`` is the Möbius coefficient m(T_t) of ``interactions[t]``, and
    v(S) = sum over T ⊆ S of m(T). Concrete subclasses decide how the sets are stored.
    """

    # Re-declared because ndlike 0.3.0 only sees a class's own annotations; see BasisGame.
    values: Float[jax.Array, "*batch n_interactions"] = field()
    players: Shaped[FinitePlayers[P], "#*batch"] = field()

    @override
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.players.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_interactions"]:
        """Return whether every interaction T is contained in S."""
        # T ⊆ S iff no player of T is missing from S.
        return ~jnp.any(self.interactions & ~coalition[..., None, :], axis=-1)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SetMoebiusGame[P, V: JaxLike](MoebiusGame[P, V]):
    """A batch of k-additive games with Möbius coefficients on explicitly stored sets.

    Sets not in ``interactions`` have coefficient zero.
    """

    # Re-declared because ndlike 0.3.0 only sees a class's own annotations; see BasisGame.
    values: Float[jax.Array, "*batch n_interactions"] = field()
    interactions: Bool[jax.Array, "n_interactions {players.n_players}"] = field()
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    additivity: int = field()
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "interactions": 2, "players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players", "interactions"}


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class AdditiveMoebiusGame[P, V: JaxLike](MoebiusGame[P, V], Additive):
    """A batch of additive games: ``values[..., i]`` is the Möbius coefficient of player position ``i``.

    The interactions are the singletons in position order and are not stored.
    """

    # Re-declared because ndlike 0.3.0 only sees a class's own annotations; see BasisGame.
    values: Float[jax.Array, "*batch {players.n_players}"] = field()
    players: Shaped[FinitePlayers[P], "#*batch"] = field()

    @property
    @override
    def interactions(self) -> Bool[jax.Array, "n_players n_players"]:
        """Return the singletons, one per player position."""
        return jnp.eye(self.players.n_players, dtype=bool)

    @property
    @override
    def additivity(self) -> Literal[1]:
        """Return the singleton interaction-order bound."""
        return 1

    @override
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.players.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_players"]:
        """Return player membership, which is containment for singletons."""
        return coalition
